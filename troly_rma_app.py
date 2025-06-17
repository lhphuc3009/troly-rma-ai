import streamlit as st
name, authentication_status, username = authenticator.login(location="main", form_name="Đăng nhập")
if authentication_status:
    import os
    from dotenv import load_dotenv
    load_dotenv()

    import pandas as pd
    import os
    import unicodedata
    import re
    import io
    import json
    import requests
    import io
    from openai import OpenAI
    from rma_ai import query_openai
    import rma_query_templates
    from rma_utils import clean_text, find_col, normalize_for_match, match_block, ensure_time_columns, extract_time_filter_from_question, filter_df_by_time
    # Đọc ánh xạ tên cột từ file JSON
    def load_column_mapping(path="uploaded_files/column_mapping.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    COLUMN_MAPPING = load_column_mapping()

    st.set_page_config(page_title="Tra cứu RMA", layout="wide")
    st.title("🔎 Tra cứu dữ liệu bảo hành - sửa chữa")

    UPLOAD_FOLDER = "uploaded_files"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    DATA_CACHE_PATH = os.path.join(UPLOAD_FOLDER, "rma_cache_data.parquet")


    def bo_loc_da_nang(df):
        df_filtered = df.copy()

        with st.sidebar.expander("🧰 Bộ lọc nâng cao", expanded=True):
            # === HÀNG 1: Năm và Tháng ===
            col1, col2 = st.columns(2)
            years = sorted(df["Năm"].dropna().unique())
            months = sorted(df["Tháng"].dropna().unique())
            selected_years = col1.multiselect(
                "Năm", years, placeholder="Chọn năm"
            )
            selected_months = col2.multiselect(
                "Tháng", months, placeholder="Chọn tháng"
            )

            # === HÀNG 2: Quý và Ngày tiếp nhận ===
            col3, col4 = st.columns(2)
            quarters = sorted(df["Quý"].dropna().unique())
            selected_quarters = col3.multiselect(
                "Quý", quarters, placeholder="Chọn quý"
            )
            date_range = col4.date_input(
                "Ngày tiếp nhận (Từ – Đến)", [], format="YYYY-MM-DD"
            )

        # Áp dụng bộ lọc
        if selected_years:
            df_filtered = df_filtered[df_filtered["Năm"].isin(selected_years)]
        if selected_months:
            df_filtered = df_filtered[df_filtered["Tháng"].isin(selected_months)]
        if selected_quarters:
            df_filtered = df_filtered[df_filtered["Quý"].isin(selected_quarters)]
        if isinstance(date_range, list) and len(date_range) == 2:
            col_date = find_col(df.columns, "ngày tiếp nhận")
            if col_date:
                df_filtered = df_filtered[
                    (df_filtered[col_date] >= pd.to_datetime(date_range[0])) &
                    (df_filtered[col_date] <= pd.to_datetime(date_range[1]))
                ]

        return df_filtered, {
            "năm": selected_years,
            "tháng": selected_months,
            "quý": selected_quarters,
            "ngày": date_range
        }

    def show_table(title, df, highlight_cols=None, key=None):
        st.markdown(f"### {title}")
        if highlight_cols is not None and len(df) > 0:
            valid_cols = [col for col in highlight_cols if col in df.columns]
            def color_green(val):
                return "color: #16A34A; font-weight: bold" if isinstance(val, (int, float)) else ""
            if "Tỷ lệ sửa thành công (%)" in df.columns:
                styled = df.style.format(
                    {"Tỷ lệ sửa thành công (%)": "{:.2f}"}
                ).applymap(color_green, subset=valid_cols)
            elif valid_cols:
                styled = df.style.applymap(color_green, subset=valid_cols)
            else:
                styled = df.style
            st.dataframe(styled)
        else:
            st.dataframe(df)
        if len(df) > 0:
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, engine='openpyxl')
            st.download_button(
                label="⬇️ Tải Excel bảng này",
                data=buffer.getvalue(),
                file_name="ket_qua_truy_van.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=key
            )

    # --- Tìm gần đúng khách hàng: block từ liền kề ---
    def search_customers_by_keyword(df, keyword):
        customer_col = find_col(df.columns, "tên khách hàng")
        if customer_col is None or not keyword:
            return df, None
        matched_names = sorted({
            val for val in df[customer_col].unique()
            if match_block(val, keyword)
        })
        df2 = df[df[customer_col].apply(lambda x: match_block(x, keyword))]
        if matched_names:
            matched_str = ", ".join(matched_names)
            return df2, matched_str
        else:
            return df, None

    def get_top_products(df, question=None):
        product_col = find_col(df.columns, "sản phẩm")
        if product_col is None:
            return pd.DataFrame()
        years, months, quarters = extract_time_filter_from_question(question or "")
        df2 = filter_df_by_time(df, years, months, quarters)
        if len(df2) > 30:
            top = df2[product_col].value_counts().head(10)
            return top.reset_index().rename(columns={"index": "Sản phẩm", product_col: "Số lượng"})
        else:
            return df2[[product_col]].value_counts().reset_index(name="Số lượng").head(10)

    def get_top_technicians(df, question=None):
        tech_col = find_col(df.columns, "kỹ thuật viên")
        if tech_col is None:
            return pd.DataFrame()
        df2 = df[df[tech_col].notna() & (df[tech_col] != "nan") & (df[tech_col].astype(str).str.strip() != "")]
        years, months, quarters = extract_time_filter_from_question(question or "")
        df2 = filter_df_by_time(df2, years, months, quarters)
        if len(df2) > 30:
            top = df2[tech_col].value_counts().head(10)
            return top.reset_index().rename(columns={"index": "Kỹ thuật viên", tech_col: "Số lượng"})
        else:
            return df2[[tech_col]].value_counts().reset_index(name="Số lượng").head(10)

    def get_top_customers(df, question=None):
        customer_col = find_col(df.columns, "tên khách hàng")
        if customer_col is None:
            return pd.DataFrame()
        years, months, quarters = extract_time_filter_from_question(question or "")
        df2 = filter_df_by_time(df, years, months, quarters)
        if len(df2) > 30:
            top = df2[customer_col].value_counts().head(10)
            return top.reset_index().rename(columns={"index": "Khách hàng", customer_col: "Số lượng"})
        else:
            return df2[[customer_col]].value_counts().reset_index(name="Số lượng").head(10)

    def extract_short_customer_from_question(q):
        m = re.search(r"([a-zA-Z0-9À-ỹ\s\-\.]+)\sgửi", q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input("🔑 Nhập OpenAI API Key của bạn", type="password")
    st.sidebar.header("🗂️ Quản lý dữ liệu")

    GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1fWFLZWyCAXn_B8jcZ0oY4KhJ8krbLPsH/export?format=csv"

    @st.cache_data
    def read_google_sheet(url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
                df.columns = [col.strip() for col in df.columns]
                return df
            else:
                st.error("Không thể tải dữ liệu từ Google Sheet.")
                return None
        except Exception as e:
            st.error(f"Lỗi khi đọc Google Sheet: {e}")
            return None

    # --- NẠP DỮ LIỆU CHÍNH ---
    data = read_google_sheet(GOOGLE_SHEET_URL)

    if data is not None:
        data["Nguồn file"] = "Google Sheet"
        data = ensure_time_columns(data)
        st.success(f"✅ Đã tải {len(data)} dòng dữ liệu từ Google Sheet.")
        data_filtered, filter_info = bo_loc_da_nang(data)
    else:
        st.warning("Không có dữ liệu nào được nạp.")
        data_filtered = None

    # === GỢI Ý KHÁCH HÀNG TỪ DỮ LIỆU ===
    st.subheader("🔍 Tìm kiếm khách hàng")

    customer_col = find_col(data_filtered.columns, "tên khách hàng")
    if customer_col:
        all_customers = sorted(data_filtered[customer_col].dropna().unique())

        if "selected_customer" not in st.session_state:
            st.session_state.selected_customer = "--"

        selected_customer = st.selectbox(
            "Chọn hoặc nhập tên khách hàng:",
            options=["--"] + all_customers,
            index=(["--"] + all_customers).index(st.session_state.selected_customer)
            if st.session_state.selected_customer in all_customers else 0
        )

        # Lưu vào session_state
        st.session_state.selected_customer = selected_customer

        if selected_customer != "--":
            df_filtered_customer = data_filtered[data_filtered[customer_col] == selected_customer]
            st.success(f"🔎 Đã lọc: {len(df_filtered_customer)} dòng cho khách hàng '{selected_customer}'")
            st.dataframe(df_filtered_customer.head(30))

        # =========== BƯỚC 3: HỎI ĐÁP AI ===============
        st.markdown("## 🤖 Hỏi dữ liệu bằng AI")
        ai_model = st.sidebar.selectbox(
            "Chọn mô hình AI",
            ["gpt-3.5-turbo", "gpt-4o"],
            index=0
        )
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        user_question = st.text_input(
            "Nhập câu hỏi về dữ liệu bảo hành/sửa chữa (bằng tiếng Việt)...",
            key="rma_ai_question"
        )
        send_btn = st.button("💬 Gửi câu hỏi", key="rma_ai_send_btn")

        if send_btn and user_question:
            q = user_question.lower()
            customer_short = extract_short_customer_from_question(q)
            summary = None
            matched_names = None

            if (
                ("tổng" in q or "tong" in q or "bao nhiêu" in q)
                and ("sản phẩm" in q or "san pham" in q)
                and not customer_short
            ):
                product_col = find_col(data_filtered.columns, "sản phẩm")
                years, months, quarters = extract_time_filter_from_question(user_question)
                df2 = filter_df_by_time(data_filtered, years, months, quarters)
                total = df2[product_col].notna().sum()
                year_str = f" trong năm {years[0]}" if years else ""
                st.success(f"Trong{year_str}, tổng cộng có {total:,} sản phẩm được bảo hành.")
                summary = df2[product_col].value_counts().reset_index().rename(
                    columns={"index": "Sản phẩm", product_col: "Số lượng"}
                ).head(10)
                show_table(
                    "Bảng tóm tắt top sản phẩm",
                    summary,
                    highlight_cols=["Số lượng"],
                    key="download_button_latest"
                )
                st.stop()
            if (
                ("tổng" in q or "tong" in q or "bao nhiêu" in q)
                and ("sản phẩm" in q or "san pham" in q)
                and customer_short
            ):
                df_customer, matched_names = search_customers_by_keyword(data_filtered, customer_short)
                product_col = find_col(df_customer.columns, "sản phẩm")
                years, months, quarters = extract_time_filter_from_question(user_question)
                df2 = filter_df_by_time(df_customer, years, months, quarters)
                total = df2[product_col].notna().sum()
                year_str = f" trong năm {years[0]}" if years else ""
                matched_note = f" (dò gần đúng: {matched_names})" if matched_names else ""
                st.success(
                    f"Trong{year_str}, Công ty {customer_short.title()}{matched_note} đã gửi tổng cộng {total:,} sản phẩm."
                )
                summary = df2[product_col].value_counts().reset_index().rename(
                    columns={"index": "Sản phẩm", product_col: "Số lượng"}
                ).head(10)
                show_table(
                    "Bảng tóm tắt top sản phẩm",
                    summary,
                    highlight_cols=["Số lượng"],
                    key="download_button_latest"
                )
                st.stop()
            if customer_short:
                df_customer, matched_names = search_customers_by_keyword(data_filtered, customer_short)
                if "sản phẩm" in q or "model" in q:
                    summary = get_top_products(df_customer, user_question)
                elif "lỗi" in q:
                    summary = pd.DataFrame()
                else:
                    summary = get_top_products(df_customer, user_question)
            elif "khách hàng" in q:
                summary = get_top_customers(data_filtered, user_question)
            elif "kỹ thuật viên" in q or "ktv" in q:
                summary = get_top_technicians(data_filtered, user_question)
            elif "sản phẩm" in q or "model" in q:
                summary = get_top_products(data_filtered, user_question)
            else:
                summary = data_filtered.head(10)
            if summary is not None and not summary.empty:
                summary = summary.reset_index(drop=True)
                if "Sản phẩm" in summary.columns and "Số lượng" in summary.columns:
                    summary = summary[["Sản phẩm", "Số lượng"]].head(10)
                elif "Model" in summary.columns and "Số lượng" in summary.columns:
                    summary = summary[["Model", "Số lượng"]].head(10)
                else:
                    summary = summary.iloc[:, :2].head(10)
                summary = summary.fillna("").astype(str)
                for col in summary.columns:
                    summary[col] = summary[col].apply(lambda x: x[:40])
                csv_data = summary.to_csv(index=False)
                if len(csv_data) > 1500:
                    st.error("❗ Kết quả quá lớn (token), vui lòng lọc thêm thời gian hoặc hỏi cụ thể hơn.")
                    st.stop()
            else:
                st.error("❗ Không tìm thấy khách hàng hoặc không có dữ liệu phù hợp!")
                st.stop()
            extra_info = ""
            if matched_names:
                extra_info = f"(Đã dò gần đúng tên khách hàng: {matched_names})\n"


            with st.spinner(f"🧠 Đang gửi dữ liệu cho AI..."):
                ai_answer, used_prompt = query_openai(
                    user_question=user_question,
                    df_summary=summary,
                    api_key=OPENAI_API_KEY,
                    model=ai_model,
                    matched_names=matched_names
                )
            st.session_state.chat_history = (st.session_state.chat_history + [{
                "q": user_question,
                "a": ai_answer,
                "tbl": summary
            }])[-5:]
            st.success("AI trả lời:")
            st.markdown(f"> {ai_answer}")
            show_table(
                "Bảng tóm tắt",
                summary,
                highlight_cols=["Số lượng"],
                key="download_button_latest"
            )


    # ========== Dưới đây là MENU TRUY VẤN NHANH SỬ DỤNG MODULE ==========
    st.sidebar.subheader("⚡ Chọn mẫu truy vấn nhanh")

    truyvan_options = [
        "Tổng số sản phẩm tiếp nhận theo tháng/năm/quý",
        "Tỷ lệ sửa chữa thành công theo tháng/năm/quý",
        "Danh sách sản phẩm chưa sửa xong trong khoảng thời gian",
        "Top 5 khách hàng gửi nhiều sản phẩm nhất",
        "Khách hàng gửi sản phẩm gì nhiều nhất",
        "Tổng sản phẩm khách hàng gửi trong năm/quý/tháng",
        "Top 5 sản phẩm bảo hành nhiều nhất",
        "Top sản phẩm thường bị từ chối bảo hành",
        "Sản phẩm có bao nhiêu lượt sửa thành công/không sửa được",
        "Top 5 lỗi kỹ thuật thường gặp nhất",
        "Lỗi nào thường gặp nhất với sản phẩm X",
        "Khách hàng thường gặp lỗi gì nhất với sản phẩm Y",
        "Thống kê số lượng đã sửa xong / không sửa được / từ chối bảo hành",
        "Tỷ lệ phần trăm sửa thành công trên tổng số tiếp nhận",
        "Danh sách sản phẩm bị từ chối bảo hành trong tháng/năm",
        "Top 3 khách hàng gửi sản phẩm X nhiều nhất trong năm",
        "Top 5 lỗi phát sinh ở khách hàng B trong quý",
        "Tỷ lệ sửa thành công sản phẩm X của khách hàng Y trong tháng",
        "Top kỹ thuật viên xử lý nhiều sản phẩm nhất",
        "Tỷ lệ sửa thành công của kỹ thuật viên theo năm/quý/tháng",
        "Thống kê số lượng sản phẩm mỗi kỹ thuật viên đã xử lý, chia theo trạng thái (đã sửa xong/không sửa được/từ chối bảo hành)"
    ]

    selected_query = st.sidebar.selectbox("Chọn truy vấn", ["-- Không chọn --"] + truyvan_options)
    st.subheader("📊 Kết quả truy vấn:")

    if selected_query == truyvan_options[0]:
        group_by = st.sidebar.selectbox("Nhóm theo", ["Tháng", "Năm", "Quý"])
        if group_by != "--":
            title, df_out = rma_query_templates.query_1_total_by_group(data_filtered, group_by)
            show_table(title, df_out, highlight_cols=["Số lượng"])

    elif selected_query == truyvan_options[1]:
        group_by = st.sidebar.selectbox("Nhóm theo", ["Tháng", "Năm", "Quý"])
        if group_by != "--":
            title, df_out = rma_query_templates.query_2_success_rate_by_group(data_filtered, group_by)
            show_table(title, df_out, highlight_cols=["Tỷ lệ sửa thành công (%)"])

    elif selected_query == truyvan_options[2]:
        title, df_out = rma_query_templates.query_3_unrepaired_products(data_filtered)
        show_table(title, df_out)

    elif selected_query == truyvan_options[3]:
        title, df_out = rma_query_templates.query_4_top_customers(data_filtered)
        show_table(title, df_out, highlight_cols=["Số lượng"])

    elif selected_query == truyvan_options[4]:
        kh = st.sidebar.selectbox("Chọn khách hàng", data_filtered[find_col(data_filtered.columns, "khách hàng")].dropna().unique())
        if kh != "--":
            title, df_out = rma_query_templates.query_5_top_products_by_customer(data_filtered, kh)
            show_table(title, df_out, highlight_cols=["Số lượng"])

        truyvan_options = [
            "Tổng số sản phẩm tiếp nhận theo tháng/năm/quý",
            "Tỷ lệ sửa chữa thành công theo tháng/năm/quý",
            "Danh sách sản phẩm chưa sửa xong trong khoảng thời gian",
            "Top 5 khách hàng gửi nhiều sản phẩm nhất",
            "Khách hàng gửi sản phẩm gì nhiều nhất",
            "Tổng sản phẩm khách hàng gửi trong năm/quý/tháng",
            "Top 5 sản phẩm bảo hành nhiều nhất",
            "Top sản phẩm thường bị từ chối bảo hành",
            "Sản phẩm có bao nhiêu lượt sửa thành công/không sửa được",
            "Top 5 lỗi kỹ thuật thường gặp nhất",
            "Lỗi nào thường gặp nhất với sản phẩm X",
            "Khách hàng thường gặp lỗi gì nhất với sản phẩm Y",
            "Thống kê số lượng đã sửa xong / không sửa được / từ chối bảo hành",
            "Tỷ lệ phần trăm sửa thành công trên tổng số tiếp nhận",
            "Danh sách sản phẩm bị từ chối bảo hành trong tháng/năm",
            "Top 3 khách hàng gửi sản phẩm X nhiều nhất trong năm",
            "Top 5 lỗi phát sinh ở khách hàng B trong quý",
            "Tỷ lệ sửa thành công sản phẩm X của khách hàng Y trong tháng",
            "Top kỹ thuật viên xử lý nhiều sản phẩm nhất",
            "Tỷ lệ sửa thành công của kỹ thuật viên theo năm/quý/tháng",
            "Thống kê số lượng sản phẩm mỗi kỹ thuật viên đã xử lý, chia theo trạng thái (đã sửa xong/không sửa được/từ chối bảo hành)"
        ]
        st.sidebar.subheader("⚡ Chọn mẫu truy vấn nhanh")
        selected_query = st.sidebar.selectbox("Chọn truy vấn", ["-- Không chọn --"] + truyvan_options)

        # ================== TRUY VẤN DÙNG data_filtered ===================
        if selected_query != "-- Không chọn --":
            customer_col = find_col(data_filtered.columns, "khach hang")
            product_col  = find_col(data_filtered.columns, "san pham")
            error_col    = find_col(data_filtered.columns, "ten loi")
            ok_col       = find_col(data_filtered.columns, "da sua xong")
            fail_col     = find_col(data_filtered.columns, "khong sua duoc")
            tcbh_col     = find_col(data_filtered.columns, "tu choi bao hanh")
            tech_col     = find_col(data_filtered.columns, "ky thuat vien")
            date_col     = find_col(data_filtered.columns, "ngay tiep nhan")

            def sidebar_selectbox(label, options):
                return st.sidebar.selectbox(label, ["--"] + list(options))
            def sidebar_multiselect(label, options):
                return st.sidebar.multiselect(label, list(options))

            st.subheader("📊 Kết quả truy vấn:")

            # 1. Tổng số sản phẩm tiếp nhận theo tháng/năm/quý
            if selected_query == truyvan_options[0]:
                group_by = sidebar_selectbox("Nhóm theo", ["Tháng", "Năm", "Quý"])
                if group_by != "--":
                    count_df = data_filtered.groupby(group_by).size().reset_index(name="Số lượng")
                    show_table(f"Tổng số sản phẩm tiếp nhận theo {group_by.lower()}", count_df, highlight_cols=["Số lượng"])

            # 2. Tỷ lệ sửa chữa thành công theo tháng/năm/quý
            elif selected_query == truyvan_options[1]:
                group_by = sidebar_selectbox("Nhóm theo", ["Tháng", "Năm", "Quý"])
                if group_by != "--" and ok_col and fail_col and tcbh_col:
                    df2 = data_filtered.copy()
                    df2["OK"] = (df2[ok_col] == 1).astype(int)
                    df2["FAIL"] = (df2[fail_col] == 1).astype(int)
                    df2["TCBH"] = (df2[tcbh_col] == 1).astype(int)
                    g = df2.groupby(group_by).agg(
                        ok = ("OK", "sum"),
                        fail = ("FAIL", "sum"),
                        tcbh = ("TCBH", "sum"),
                    ).reset_index()
                    g["Tỷ lệ sửa thành công (%)"] = round(g["ok"] / (g["ok"] + g["fail"] + g["tcbh"]) * 100, 2)
                    show_table(f"Tỷ lệ sửa chữa thành công theo {group_by.lower()}", g[[group_by, "ok", "fail", "tcbh", "Tỷ lệ sửa thành công (%)"]], highlight_cols=["Tỷ lệ sửa thành công (%)"])

            # 3. Danh sách sản phẩm chưa sửa xong trong khoảng thời gian
            elif selected_query == truyvan_options[2]:
                if date_col and ok_col:
                    df3 = data_filtered[data_filtered[ok_col] != 1]
                    show_table("Danh sách sản phẩm chưa sửa xong", df3[[date_col, customer_col, product_col, ok_col]])
                else:
                    st.error("Không tìm thấy cột ngày hoặc trạng thái 'Đã sửa xong'!")

            # 4. Top 5 khách hàng gửi nhiều sản phẩm nhất
            elif selected_query == truyvan_options[3]:
                top_kh = data_filtered[customer_col].value_counts().head(5)
                df_kh = pd.DataFrame({"Khách hàng": top_kh.index, "Số lượng": top_kh.values})
                show_table("Top 5 khách hàng gửi nhiều sản phẩm nhất", df_kh, highlight_cols=["Số lượng"])

            # 5. Khách hàng gửi sản phẩm gì nhiều nhất
            elif selected_query == truyvan_options[4]:
                kh = sidebar_selectbox("Chọn khách hàng", data_filtered[customer_col].dropna().unique())
                if kh != "--":
                    top_sp = data_filtered[data_filtered[customer_col] == kh][product_col].value_counts().head(5)
                    df_sp = pd.DataFrame({"Sản phẩm": top_sp.index, "Số lượng": top_sp.values})
                    show_table(f"Top sản phẩm khách hàng {kh} đã gửi", df_sp, highlight_cols=["Số lượng"])

            # 6. Tổng sản phẩm khách hàng gửi trong năm/quý/tháng
            elif selected_query == truyvan_options[5]:
                kh = sidebar_selectbox("Chọn khách hàng", data_filtered[customer_col].dropna().unique())
                group_by = sidebar_selectbox("Nhóm theo", ["Năm", "Quý", "Tháng"])
                if kh != "--" and group_by != "--":
                    df6 = data_filtered[data_filtered[customer_col] == kh]
                    count_df = df6.groupby(group_by).size().reset_index(name="Số lượng")
                    show_table(f"Tổng sản phẩm khách hàng {kh} gửi theo {group_by.lower()}", count_df, highlight_cols=["Số lượng"])

            # 7. Top 5 sản phẩm bảo hành nhiều nhất
            elif selected_query == truyvan_options[6]:
                top_sp = data_filtered[product_col].value_counts().head(5)
                df_sp = pd.DataFrame({"Sản phẩm": top_sp.index, "Số lượng": top_sp.values})
                show_table("Top 5 sản phẩm bảo hành nhiều nhất", df_sp, highlight_cols=["Số lượng"])

            # 8. Top sản phẩm thường bị từ chối bảo hành
            elif selected_query == truyvan_options[7]:
                if tcbh_col:
                    top_sp = data_filtered[data_filtered[tcbh_col] == 1][product_col].value_counts().head(5)
                    df_sp = pd.DataFrame({"Sản phẩm": top_sp.index, "Số lượng": top_sp.values})
                    show_table("Top sản phẩm bị từ chối bảo hành nhiều nhất", df_sp, highlight_cols=["Số lượng"])
                else:
                    st.error("Không tìm thấy cột 'Từ chối bảo hành' trong dữ liệu!")

            # 9. Sản phẩm có bao nhiêu lượt sửa thành công/không sửa được
            elif selected_query == truyvan_options[8]:
                sp = sidebar_selectbox("Chọn sản phẩm", data_filtered[product_col].dropna().unique())
                if sp != "--" and ok_col and fail_col and tcbh_col:
                    df9 = data_filtered[data_filtered[product_col] == sp]
                    ok = (df9[ok_col] == 1).sum()
                    fail = (df9[fail_col] == 1).sum()
                    tcbh = (df9[tcbh_col] == 1).sum()
                    df_stat = pd.DataFrame({
                        "Trạng thái": ["Sửa xong", "Không sửa được", "Từ chối BH"],
                        "Số lượng": [ok, fail, tcbh]
                    })
                    show_table(f"Số lượt xử lý của {sp}", df_stat, highlight_cols=["Số lượng"])
                else:
                    st.error("Không tìm thấy các cột trạng thái xử lý trong dữ liệu!")

            # 10. Top 5 lỗi kỹ thuật thường gặp nhất
            elif selected_query == truyvan_options[9]:
                if error_col:
                    top_err = data_filtered[error_col].value_counts().head(5)
                    df_err = pd.DataFrame({"Lỗi kỹ thuật": top_err.index, "Số lần": top_err.values})
                    show_table("Top 5 lỗi kỹ thuật thường gặp nhất", df_err, highlight_cols=["Số lần"])
                else:
                    st.error("Không tìm thấy cột tên lỗi!")

            # 11. Lỗi nào thường gặp nhất với sản phẩm X
            elif selected_query == truyvan_options[10]:
                sp = sidebar_selectbox("Chọn sản phẩm", data_filtered[product_col].dropna().unique())
                if sp != "--" and error_col:
                    top_err = data_filtered[data_filtered[product_col] == sp][error_col].value_counts().head(5)
                    df_err = pd.DataFrame({"Lỗi kỹ thuật": top_err.index, "Số lần": top_err.values})
                    show_table(f"Top lỗi thường gặp nhất của sản phẩm {sp}", df_err, highlight_cols=["Số lần"])

            # 12. Khách hàng thường gặp lỗi gì nhất với sản phẩm Y
            elif selected_query == truyvan_options[11]:
                kh = sidebar_selectbox("Chọn khách hàng", data_filtered[customer_col].dropna().unique())
                sp = sidebar_selectbox("Chọn sản phẩm", data_filtered[product_col].dropna().unique())
                if kh != "--" and sp != "--" and error_col:
                    top_err = data_filtered[(data_filtered[customer_col] == kh) & (data_filtered[product_col] == sp)][error_col].value_counts().head(5)
                    df_err = pd.DataFrame({"Lỗi kỹ thuật": top_err.index, "Số lần": top_err.values})
                    show_table(f"Top lỗi khách hàng {kh} gặp với {sp}", df_err, highlight_cols=["Số lần"])

            # 13. Thống kê số lượng đã sửa xong / không sửa được / từ chối bảo hành
            elif selected_query == truyvan_options[12]:
                if ok_col and fail_col and tcbh_col:
                    ok = (data_filtered[ok_col] == 1).sum()
                    fail = (data_filtered[fail_col] == 1).sum()
                    tcbh = (data_filtered[tcbh_col] == 1).sum()
                    total = ok + fail + tcbh
                    df_stat = pd.DataFrame({
                        "Trạng thái": ["Sửa xong", "Không sửa được", "Từ chối BH"],
                        "Số lượng": [ok, fail, tcbh]
                    })
                    show_table("Thống kê số lượng xử lý theo trạng thái", df_stat, highlight_cols=["Số lượng"])
                    st.markdown(f"**Tổng đã xử lý: {total}**", unsafe_allow_html=True)
                else:
                    st.error("Không tìm thấy các cột trạng thái xử lý trong dữ liệu!")

            # 14. Tỷ lệ phần trăm sửa thành công trên tổng số tiếp nhận
            elif selected_query == truyvan_options[13]:
                if ok_col and fail_col and tcbh_col:
                    ok = (data_filtered[ok_col] == 1).sum()
                    fail = (data_filtered[fail_col] == 1).sum()
                    tcbh = (data_filtered[tcbh_col] == 1).sum()
                    total = ok + fail + tcbh
                    percent = round(ok / total * 100, 2) if total > 0 else 0
                    df_percent = pd.DataFrame({"Tổng xử lý": [total], "Sửa thành công (%)": [percent]})
                    show_table("Tỷ lệ sửa thành công trên tổng số tiếp nhận", df_percent, highlight_cols=["Sửa thành công (%)"])
                else:
                    st.error("Không tìm thấy các cột trạng thái xử lý trong dữ liệu!")

            # 15. Danh sách sản phẩm bị từ chối bảo hành trong tháng/năm
            elif selected_query == truyvan_options[14]:
                if tcbh_col:
                    df15 = data_filtered[data_filtered[tcbh_col] == 1]
                    show_table("Sản phẩm bị từ chối bảo hành", df15[[product_col, customer_col, "Tháng", "Năm"]])
                else:
                    st.error("Không tìm thấy cột từ chối bảo hành!")

            # 16. Top 3 khách hàng gửi sản phẩm X nhiều nhất trong năm
            elif selected_query == truyvan_options[15]:
                sp = sidebar_selectbox("Chọn sản phẩm", data_filtered[product_col].dropna().unique())
                if sp != "--":
                    top_kh = data_filtered[data_filtered[product_col] == sp][customer_col].value_counts().head(3)
                    df_kh = pd.DataFrame({"Khách hàng": top_kh.index, "Số lượng": top_kh.values})
                    show_table(f"Top 3 khách hàng gửi {sp} nhiều nhất", df_kh, highlight_cols=["Số lượng"])

            # 17. Top 5 lỗi phát sinh ở khách hàng B trong quý
            elif selected_query == truyvan_options[16]:
                kh = sidebar_selectbox("Chọn khách hàng", data_filtered[customer_col].dropna().unique())
                q = sidebar_selectbox("Chọn quý", sorted(data_filtered["Quý"].dropna().unique()))
                if kh != "--" and q != "--" and error_col:
                    df17 = data_filtered[(data_filtered[customer_col] == kh) & (data_filtered["Quý"] == q)]
                    top_err = df17[error_col].value_counts().head(5)
                    df_err = pd.DataFrame({"Lỗi kỹ thuật": top_err.index, "Số lần": top_err.values})
                    show_table(f"Top lỗi của {kh} trong quý {q}", df_err, highlight_cols=["Số lần"])

            # 18. Tỷ lệ sửa thành công sản phẩm X của khách hàng Y trong tháng
            elif selected_query == truyvan_options[17]:
                sp = sidebar_selectbox("Chọn sản phẩm", data_filtered[product_col].dropna().unique())
                kh = sidebar_selectbox("Chọn khách hàng", data_filtered[customer_col].dropna().unique())
                th = sidebar_selectbox("Chọn tháng", sorted(data_filtered["Tháng"].dropna().unique()))
                if sp != "--" and kh != "--" and th != "--" and ok_col and fail_col and tcbh_col:
                    df18 = data_filtered[
                        (data_filtered[product_col] == sp) &
                        (data_filtered[customer_col] == kh) &
                        (data_filtered["Tháng"] == th)
                    ]
                    ok = (df18[ok_col] == 1).sum()
                    fail = (df18[fail_col] == 1).sum()
                    tcbh = (df18[tcbh_col] == 1).sum()
                    total = ok + fail + tcbh
                    percent = round(ok / total * 100, 2) if total > 0 else 0
                    df_percent = pd.DataFrame({"Tổng xử lý": [total], "Sửa thành công (%)": [percent]})
                    show_table(f"Tỷ lệ sửa thành công {sp} của {kh} trong tháng {th}", df_percent, highlight_cols=["Sửa thành công (%)"])
                else:
                    st.error("Không tìm thấy các cột trạng thái xử lý trong dữ liệu!")

            # 19. Top 5 kỹ thuật viên xử lý nhiều sản phẩm nhất
            elif selected_query == truyvan_options[18]:
                if tech_col:
                    top_ktv = data_filtered[tech_col].value_counts().head(5)
                    df_ktv = pd.DataFrame({"Kỹ thuật viên": top_ktv.index, "Số lượng": top_ktv.values})
                    show_table("Top kỹ thuật viên xử lý nhiều sản phẩm nhất", df_ktv, highlight_cols=["Số lượng"])
                else:
                    st.error("Không tìm thấy cột 'Kỹ thuật viên' trong dữ liệu!")

            # 20. Tỷ lệ sửa thành công của kỹ thuật viên theo năm/quý/tháng
            elif selected_query == truyvan_options[19]:
                group_by = sidebar_selectbox("Nhóm theo", ["Năm", "Quý", "Tháng"])
                if tech_col and ok_col and fail_col and tcbh_col and group_by != "--":
                    df20 = data_filtered.copy()
                    df20["OK"] = (df20[ok_col] == 1).astype(int)
                    df20["FAIL"] = (df20[fail_col] == 1).astype(int)
                    df20["TCBH"] = (df20[tcbh_col] == 1).astype(int)
                    group = [group_by, tech_col]
                    g = df20.groupby(group).agg(
                        ok = ("OK", "sum"),
                        fail = ("FAIL", "sum"),
                        tcbh = ("TCBH", "sum"),
                    ).reset_index()
                    g["Tỷ lệ sửa thành công (%)"] = round(g["ok"] / (g["ok"] + g["fail"] + g["tcbh"]) * 100, 2)
                    show_table(f"Tỷ lệ sửa thành công của kỹ thuật viên theo {group_by.lower()}", g[[group_by, tech_col, "ok", "fail", "tcbh", "Tỷ lệ sửa thành công (%)"]], highlight_cols=["Tỷ lệ sửa thành công (%)"])
                else:
                    st.error("Không tìm thấy cột kỹ thuật viên hoặc cột trạng thái!")

            # 21. Thống kê số lượng sản phẩm mỗi kỹ thuật viên đã xử lý, chia theo trạng thái (đã sửa xong/không sửa được/từ chối bảo hành)
            elif selected_query == truyvan_options[20]:
                if tech_col and ok_col and fail_col and tcbh_col:
                    g = data_filtered.groupby(tech_col).agg(
                        ok   = (ok_col,   lambda x: (x == 1).sum()),
                        fail = (fail_col, lambda x: (x == 1).sum()),
                        tcbh = (tcbh_col, lambda x: (x == 1).sum()),
                    ).reset_index()
                    g["Tổng xử lý"] = g["ok"] + g["fail"] + g["tcbh"]
                    g["Tỷ lệ sửa thành công (%)"] = g.apply(
                        lambda row: round(row["ok"] / row["Tổng xử lý"] * 100, 2) if row["Tổng xử lý"] > 0 else 0, axis=1
                    )
                    show_table(
                        "Thống kê số lượng sản phẩm mỗi kỹ thuật viên đã xử lý",
                        g.rename(columns={
                            "ok": "Sửa xong",
                            "fail": "Không sửa được",
                            "tcbh": "Từ chối BH",
                            "Tổng xử lý": "Tổng xử lý",
                            "Tỷ lệ sửa thành công (%)": "Tỷ lệ sửa thành công (%)"
                        }),
                        highlight_cols=[
                            "Sửa xong", "Không sửa được", "Từ chối BH",
                            "Tổng xử lý", "Tỷ lệ sửa thành công (%)"
                        ]
                    )
                else:
                    st.error("Không tìm thấy cột kỹ thuật viên hoặc cột trạng thái!")

    elif selected_query == truyvan_options[5]:
        kh = st.sidebar.selectbox("Chọn khách hàng", data_filtered[find_col(data_filtered.columns, "khách hàng")].dropna().unique())
        group_by = st.sidebar.selectbox("Nhóm theo", ["Năm", "Quý", "Tháng"])
        if kh != "--" and group_by != "--":
            title, df_out = rma_query_templates.query_6_total_by_customer_and_time(data_filtered, kh, group_by)
            show_table(title, df_out, highlight_cols=["Số lượng"])

    elif selected_query == truyvan_options[6]:
        title, df_out = rma_query_templates.query_7_top_products(data_filtered)
        show_table(title, df_out, highlight_cols=["Số lượng"])

    elif selected_query == truyvan_options[7]:
        title, df_out = rma_query_templates.query_8_top_rejected_products(data_filtered)
        show_table(title, df_out, highlight_cols=["Số lượng"])

    elif selected_query == truyvan_options[8]:
        sp = st.sidebar.selectbox("Chọn sản phẩm", data_filtered[find_col(data_filtered.columns, "sản phẩm")].dropna().unique())
        if sp != "--":
            title, df_out = rma_query_templates.query_9_product_status_counts(data_filtered, sp)
            show_table(title, df_out, highlight_cols=["Số lượng"])

    elif selected_query == truyvan_options[9]:
        title, df_out = rma_query_templates.query_10_top_errors(data_filtered)
        show_table(title, df_out, highlight_cols=["Số lần"])

    elif selected_query == truyvan_options[10]:
        sp = st.sidebar.selectbox("Chọn sản phẩm", data_filtered[find_col(data_filtered.columns, "sản phẩm")].dropna().unique())
        if sp != "--":
            title, df_out = rma_query_templates.query_11_top_errors_by_product(data_filtered, sp)
            show_table(title, df_out, highlight_cols=["Số lần"])

    elif selected_query == truyvan_options[11]:
        kh = st.sidebar.selectbox("Chọn khách hàng", data_filtered[find_col(data_filtered.columns, "khách hàng")].dropna().unique())
        sp = st.sidebar.selectbox("Chọn sản phẩm", data_filtered[find_col(data_filtered.columns, "sản phẩm")].dropna().unique())
        if kh != "--" and sp != "--":
            title, df_out = rma_query_templates.query_12_errors_by_customer_and_product(data_filtered, kh, sp)
            show_table(title, df_out, highlight_cols=["Số lần"])

    elif selected_query == truyvan_options[12]:
        title, df_out = rma_query_templates.query_13_status_summary(data_filtered)
        show_table(title, df_out, highlight_cols=["Số lượng"])

    elif selected_query == truyvan_options[13]:
        title, df_out = rma_query_templates.query_14_success_rate_overall(data_filtered)
        show_table(title, df_out, highlight_cols=["Sửa thành công (%)"])

    elif selected_query == truyvan_options[14]:
        title, df_out = rma_query_templates.query_15_rejected_products_by_time(data_filtered)
        show_table(title, df_out)

    elif selected_query == truyvan_options[15]:
        sp = st.sidebar.selectbox("Chọn sản phẩm", data_filtered[find_col(data_filtered.columns, "sản phẩm")].dropna().unique())
        if sp != "--":
            title, df_out = rma_query_templates.query_16_top_customers_by_product(data_filtered, sp)
            show_table(title, df_out, highlight_cols=["Số lượng"])

    elif selected_query == truyvan_options[16]:
        kh = st.sidebar.selectbox("Chọn khách hàng", data_filtered[find_col(data_filtered.columns, "khách hàng")].dropna().unique())
        q = st.sidebar.selectbox("Chọn quý", sorted(data_filtered["Quý"].dropna().unique()))
        if kh != "--" and q != "--":
            title, df_out = rma_query_templates.query_17_top_errors_by_customer_and_quarter(data_filtered, kh, q)
            show_table(title, df_out, highlight_cols=["Số lần"])

    elif selected_query == truyvan_options[17]:
        sp = st.sidebar.selectbox("Chọn sản phẩm", data_filtered[find_col(data_filtered.columns, "sản phẩm")].dropna().unique())
        kh = st.sidebar.selectbox("Chọn khách hàng", data_filtered[find_col(data_filtered.columns, "khách hàng")].dropna().unique())
        th = st.sidebar.selectbox("Chọn tháng", sorted(data_filtered["Tháng"].dropna().unique()))
        if sp != "--" and kh != "--" and th != "--":
            title, df_out = rma_query_templates.query_18_success_rate_by_customer_product_month(data_filtered, kh, sp, th)
            show_table(title, df_out, highlight_cols=["Sửa thành công (%)"])

    elif selected_query == truyvan_options[18]:
        title, df_out = rma_query_templates.query_19_top_technicians(data_filtered)
        show_table(title, df_out, highlight_cols=["Số lượng"])

    elif selected_query == truyvan_options[19]:
        group_by = st.sidebar.selectbox("Nhóm theo", ["Năm", "Quý", "Tháng"])
        if group_by != "--":
            title, df_out = rma_query_templates.query_20_success_rate_by_technician_and_group(data_filtered, group_by)
            show_table(title, df_out, highlight_cols=["Tỷ lệ sửa thành công (%)"])

    elif selected_query == truyvan_options[20]:
        title, df_out = rma_query_templates.query_21_technician_status_summary(data_filtered)
        show_table(title, df_out, highlight_cols=["Sửa xong", "Không sửa được", "Từ chối BH", "Tổng xử lý", "Tỷ lệ sửa thành công (%)"])