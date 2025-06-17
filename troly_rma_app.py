
import streamlit as st

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

with open("auth_config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login("Đăng nhập", "main")

if authentication_status is False:
    st.error("Sai tên đăng nhập hoặc mật khẩu")
elif authentication_status is None:
    st.warning("Vui lòng nhập thông tin đăng nhập")
elif authentication_status:
    st.sidebar.success(f"Xin chào {name}")
    authenticator.logout("Đăng xuất", "sidebar")

if authentication_status:
    import streamlit_authenticator as stauth
    import yaml
    import pandas as pd
    from yaml.loader import SafeLoader
    from rma_ai import process_ai_query
    from rma_utils import load_data_from_drive, filter_by_time_range
    from rma_query_templates import *
    
    # Load cấu hình đăng nhập từ file YAML
    with open("auth_config.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)
    
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    
    name, authentication_status, username = authenticator.login("Đăng nhập", "main")
    
    if authentication_status is False:
        st.error("Sai tên đăng nhập hoặc mật khẩu")
    elif authentication_status is None:
        st.warning("Vui lòng nhập thông tin đăng nhập")
    elif authentication_status:
        st.sidebar.success(f"Xin chào {name}")
        authenticator.logout("Đăng xuất", "sidebar")
    
        st.title("🔧 Trợ Lý Bảo Hành AI")
    
        uploaded_file = st.file_uploader("📤 Tải lên file Excel từ Google Drive (export thủ công)", type=["xlsx"])
        df = load_data_from_drive(uploaded_file) if uploaded_file else None
    
        if df is not None:
            st.subheader("📊 Truy vấn nhanh")
            query = st.selectbox("Chọn truy vấn mẫu:", [
                "Top 5 khách hàng gửi nhiều nhất",
                "Top 5 sản phẩm phổ biến nhất",
                "Tỷ lệ sửa chữa thành công",
                "Tỷ lệ không sửa được",
                "Tổng số sản phẩm đã tiếp nhận",
                "Tổng số sản phẩm đã sửa xong",
                "Tổng số sản phẩm không sửa được",
                "Tổng số sản phẩm bị từ chối bảo hành",
                "Số lượng sản phẩm theo khách hàng",
                "Số lượng sản phẩm theo nhóm hàng",
                "Sản phẩm phổ biến theo khách hàng",
                "Tổng số sản phẩm theo tháng",
                "Tổng số sản phẩm theo năm",
                "Top khách hàng theo năm",
                "Top sản phẩm theo quý",
                "Tỷ lệ sửa chữa theo khách hàng",
                "Tỷ lệ sửa chữa theo nhóm hàng",
                "Tổng số sản phẩm theo quý",
                "Sản phẩm không sửa được theo nhóm hàng",
                "Top khách hàng bị từ chối bảo hành",
                "Tổng hợp trạng thái xử lý"
            ])
    
            if st.button("Thực hiện truy vấn"):
                if "khách hàng gửi nhiều" in query:
                    st.dataframe(top_customers(df))
                elif "sản phẩm phổ biến" in query:
                    st.dataframe(top_products(df))
                elif "sửa chữa thành công" in query:
                    st.dataframe(success_rate(df))
                elif "không sửa được" in query and "tỷ lệ" in query:
                    st.dataframe(failure_rate(df))
                elif "tiếp nhận" in query:
                    st.dataframe(total_received(df))
                elif "sửa xong" in query:
                    st.dataframe(total_repaired(df))
                elif "không sửa được" in query:
                    st.dataframe(total_failed(df))
                elif "từ chối bảo hành" in query and "bị" not in query:
                    st.dataframe(total_rejected(df))
                elif "theo khách hàng" in query and "sản phẩm phổ biến" not in query:
                    st.dataframe(count_by_customer(df))
                elif "theo nhóm hàng" in query and "không sửa" not in query:
                    st.dataframe(count_by_category(df))
                elif "sản phẩm phổ biến theo khách hàng" in query:
                    st.dataframe(top_products_by_customer(df))
                elif "theo tháng" in query:
                    st.dataframe(count_by_month(df))
                elif "theo năm" in query and "khách hàng" not in query:
                    st.dataframe(count_by_year(df))
                elif "top khách hàng theo năm" in query:
                    st.dataframe(top_customers_by_year(df))
                elif "theo quý" in query and "sản phẩm" in query:
                    st.dataframe(top_products_by_quarter(df))
                elif "tỷ lệ sửa chữa theo khách hàng" in query:
                    st.dataframe(success_rate_by_customer(df))
                elif "tỷ lệ sửa chữa theo nhóm hàng" in query:
                    st.dataframe(success_rate_by_category(df))
                elif "tổng số sản phẩm theo quý" in query:
                    st.dataframe(count_by_quarter(df))
                elif "không sửa được theo nhóm hàng" in query:
                    st.dataframe(failure_by_category(df))
                elif "bị từ chối bảo hành" in query:
                    st.dataframe(rejected_by_customer(df))
                elif "tổng hợp trạng thái" in query:
                    st.dataframe(summary_status(df))
    
        st.markdown("---")
        st.subheader("🧠 Hỏi dữ liệu bằng AI")
        api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else st.text_input("Nhập OpenAI API Key", type="password")
        user_question = st.text_area("Đặt câu hỏi:")
    
        if st.button("Trả lời"):
            if api_key and user_question:
                with st.spinner("Đang xử lý..."):
                    answer = process_ai_query(df, user_question, api_key)
                    st.success("Câu trả lời:")
                    st.write(answer)
            else:
                st.warning("Vui lòng nhập đầy đủ API key và câu hỏi.")