
import pandas as pd
from rma_utils import find_col

def query_1_total_by_group(df, group_by):
    count_df = df.groupby(group_by).size().reset_index(name="Số lượng")
    return f"Tổng số sản phẩm tiếp nhận theo {group_by.lower()}", count_df

def query_2_success_rate_by_group(df, group_by):
    ok_col = find_col(df.columns, "đã sửa xong")
    fail_col = find_col(df.columns, "không sửa được")
    tcbh_col = find_col(df.columns, "từ chối bảo hành")
    if not all([ok_col, fail_col, tcbh_col]):
        return "Không đủ cột trạng thái!", pd.DataFrame()
    df2 = df.copy()
    df2["OK"] = (df2[ok_col] == 1).astype(int)
    df2["FAIL"] = (df2[fail_col] == 1).astype(int)
    df2["TCBH"] = (df2[tcbh_col] == 1).astype(int)
    g = df2.groupby(group_by).agg(
        ok=("OK", "sum"),
        fail=("FAIL", "sum"),
        tcbh=("TCBH", "sum"),
    ).reset_index()
    g["Tỷ lệ sửa thành công (%)"] = round(g["ok"] / (g["ok"] + g["fail"] + g["tcbh"]) * 100, 2)
    return f"Tỷ lệ sửa chữa thành công theo {group_by.lower()}", g[[group_by, "ok", "fail", "tcbh", "Tỷ lệ sửa thành công (%)"]]

def query_3_unrepaired_products(df):
    ok_col = find_col(df.columns, "đã sửa xong")
    date_col = find_col(df.columns, "ngày tiếp nhận")
    customer_col = find_col(df.columns, "khách hàng")
    product_col = find_col(df.columns, "sản phẩm")
    if not all([ok_col, date_col]):
        return "Thiếu cột trạng thái hoặc ngày!", pd.DataFrame()
    df3 = df[df[ok_col] != 1]
    return "Danh sách sản phẩm chưa sửa xong", df3[[date_col, customer_col, product_col, ok_col]]

def query_4_top_customers(df, top_n=5):
    customer_col = find_col(df.columns, "khách hàng")
    if not customer_col:
        return "Không có cột 'Khách hàng'", pd.DataFrame()
    top_kh = df[customer_col].value_counts().head(top_n)
    return f"Top {top_n} khách hàng gửi nhiều sản phẩm nhất", pd.DataFrame({"Khách hàng": top_kh.index, "Số lượng": top_kh.values})

def query_5_top_products_by_customer(df, customer_name, top_n=5):
    customer_col = find_col(df.columns, "khách hàng")
    product_col = find_col(df.columns, "sản phẩm")
    if not customer_col or not product_col:
        return "Thiếu cột khách hàng hoặc sản phẩm!", pd.DataFrame()
    df_filtered = df[df[customer_col] == customer_name]
    top_sp = df_filtered[product_col].value_counts().head(top_n)
    return f"Top sản phẩm khách hàng {customer_name} đã gửi", pd.DataFrame({"Sản phẩm": top_sp.index, "Số lượng": top_sp.values})

# Placeholder cho truy vấn 6 đến 21 (để tránh quá tải trong 1 lần chạy)
def query_6_to_21_placeholder():
    return "Các truy vấn từ 6 đến 21 đang được hoàn thiện...", pd.DataFrame()

def query_6_total_by_customer_and_time(df, customer_name, group_by):
    customer_col = find_col(df.columns, "khách hàng")
    if customer_col is None or group_by not in df.columns:
        return "Thiếu cột khách hàng hoặc nhóm thời gian!", pd.DataFrame()
    df_filtered = df[df[customer_col] == customer_name]
    result = df_filtered.groupby(group_by).size().reset_index(name="Số lượng")
    return f"Tổng sản phẩm khách hàng {customer_name} gửi theo {group_by.lower()}", result

def query_7_top_products(df, top_n=5):
    product_col = find_col(df.columns, "sản phẩm")
    if product_col is None:
        return "Không tìm thấy cột sản phẩm!", pd.DataFrame()
    top = df[product_col].value_counts().head(top_n)
    return f"Top {top_n} sản phẩm bảo hành nhiều nhất", pd.DataFrame({"Sản phẩm": top.index, "Số lượng": top.values})

def query_8_top_rejected_products(df, top_n=5):
    product_col = find_col(df.columns, "sản phẩm")
    tcbh_col = find_col(df.columns, "từ chối bảo hành")
    if not all([product_col, tcbh_col]):
        return "Thiếu cột sản phẩm hoặc từ chối bảo hành!", pd.DataFrame()
    top = df[df[tcbh_col] == 1][product_col].value_counts().head(top_n)
    return "Top sản phẩm bị từ chối bảo hành nhiều nhất", pd.DataFrame({"Sản phẩm": top.index, "Số lượng": top.values})

def query_9_product_status_counts(df, product_name):
    product_col = find_col(df.columns, "sản phẩm")
    ok_col = find_col(df.columns, "đã sửa xong")
    fail_col = find_col(df.columns, "không sửa được")
    tcbh_col = find_col(df.columns, "từ chối bảo hành")
    if not all([product_col, ok_col, fail_col, tcbh_col]):
        return "Thiếu các cột xử lý sản phẩm!", pd.DataFrame()
    df_filtered = df[df[product_col] == product_name]
    ok = (df_filtered[ok_col] == 1).sum()
    fail = (df_filtered[fail_col] == 1).sum()
    tcbh = (df_filtered[tcbh_col] == 1).sum()
    result = pd.DataFrame({
        "Trạng thái": ["Sửa xong", "Không sửa được", "Từ chối BH"],
        "Số lượng": [ok, fail, tcbh]
    })
    return f"Số lượt xử lý của {product_name}", result

def query_10_top_errors(df, top_n=5):
    error_col = find_col(df.columns, "tên lỗi")
    if not error_col:
        return "Không có cột tên lỗi!", pd.DataFrame()
    top = df[error_col].value_counts().head(top_n)
    return "Top lỗi kỹ thuật thường gặp nhất", pd.DataFrame({"Lỗi kỹ thuật": top.index, "Số lần": top.values})

def query_11_top_errors_by_product(df, product_name, top_n=5):
    product_col = find_col(df.columns, "sản phẩm")
    error_col = find_col(df.columns, "tên lỗi")
    if not product_col or not error_col:
        return "Thiếu cột sản phẩm hoặc tên lỗi!", pd.DataFrame()
    top = df[df[product_col] == product_name][error_col].value_counts().head(top_n)
    return f"Top lỗi thường gặp nhất của sản phẩm {product_name}", pd.DataFrame({"Lỗi kỹ thuật": top.index, "Số lần": top.values})

def query_12_errors_by_customer_and_product(df, customer_name, product_name, top_n=5):
    customer_col = find_col(df.columns, "khách hàng")
    product_col = find_col(df.columns, "sản phẩm")
    error_col = find_col(df.columns, "tên lỗi")
    if not all([customer_col, product_col, error_col]):
        return "Thiếu cột khách hàng, sản phẩm hoặc lỗi!", pd.DataFrame()
    df_filtered = df[(df[customer_col] == customer_name) & (df[product_col] == product_name)]
    top = df_filtered[error_col].value_counts().head(top_n)
    return f"Top lỗi khách hàng {customer_name} gặp với {product_name}", pd.DataFrame({"Lỗi kỹ thuật": top.index, "Số lần": top.values})

def query_13_status_summary(df):
    ok_col = find_col(df.columns, "đã sửa xong")
    fail_col = find_col(df.columns, "không sửa được")
    tcbh_col = find_col(df.columns, "từ chối bảo hành")
    if not all([ok_col, fail_col, tcbh_col]):
        return "Thiếu cột trạng thái xử lý!", pd.DataFrame()
    ok = (df[ok_col] == 1).sum()
    fail = (df[fail_col] == 1).sum()
    tcbh = (df[tcbh_col] == 1).sum()
    result = pd.DataFrame({
        "Trạng thái": ["Sửa xong", "Không sửa được", "Từ chối BH"],
        "Số lượng": [ok, fail, tcbh]
    })
    return "Thống kê số lượng xử lý theo trạng thái", result

def query_14_success_rate_overall(df):
    ok_col = find_col(df.columns, "đã sửa xong")
    fail_col = find_col(df.columns, "không sửa được")
    tcbh_col = find_col(df.columns, "từ chối bảo hành")
    if not all([ok_col, fail_col, tcbh_col]):
        return "Thiếu cột trạng thái!", pd.DataFrame()
    ok = (df[ok_col] == 1).sum()
    fail = (df[fail_col] == 1).sum()
    tcbh = (df[tcbh_col] == 1).sum()
    total = ok + fail + tcbh
    percent = round(ok / total * 100, 2) if total > 0 else 0
    return "Tỷ lệ sửa thành công trên tổng số tiếp nhận", pd.DataFrame({"Tổng xử lý": [total], "Sửa thành công (%)": [percent]})

# Các truy vấn 15–21 sẽ tiếp tục ở bước sau nếu cần

def query_15_rejected_products_by_time(df):
    product_col = find_col(df.columns, "sản phẩm")
    customer_col = find_col(df.columns, "khách hàng")
    tcbh_col = find_col(df.columns, "từ chối bảo hành")
    if not all([product_col, customer_col, tcbh_col]):
        return "Thiếu cột cần thiết!", pd.DataFrame()
    df15 = df[df[tcbh_col] == 1]
    return "Sản phẩm bị từ chối bảo hành", df15[[product_col, customer_col, "Tháng", "Năm"]]

def query_16_top_customers_by_product(df, product_name, top_n=3):
    product_col = find_col(df.columns, "sản phẩm")
    customer_col = find_col(df.columns, "khách hàng")
    if not all([product_col, customer_col]):
        return "Thiếu cột sản phẩm hoặc khách hàng!", pd.DataFrame()
    top_kh = df[df[product_col] == product_name][customer_col].value_counts().head(top_n)
    return f"Top {top_n} khách hàng gửi {product_name} nhiều nhất", pd.DataFrame({"Khách hàng": top_kh.index, "Số lượng": top_kh.values})

def query_17_top_errors_by_customer_and_quarter(df, customer_name, quarter):
    customer_col = find_col(df.columns, "khách hàng")
    error_col = find_col(df.columns, "tên lỗi")
    if not all([customer_col, error_col]):
        return "Thiếu cột khách hàng hoặc lỗi!", pd.DataFrame()
    df_filtered = df[(df[customer_col] == customer_name) & (df["Quý"] == quarter)]
    top = df_filtered[error_col].value_counts().head(5)
    return f"Top lỗi của {customer_name} trong quý {quarter}", pd.DataFrame({"Lỗi kỹ thuật": top.index, "Số lần": top.values})

def query_18_success_rate_by_customer_product_month(df, customer_name, product_name, month):
    customer_col = find_col(df.columns, "khách hàng")
    product_col = find_col(df.columns, "sản phẩm")
    ok_col = find_col(df.columns, "đã sửa xong")
    fail_col = find_col(df.columns, "không sửa được")
    tcbh_col = find_col(df.columns, "từ chối bảo hành")
    if not all([customer_col, product_col, ok_col, fail_col, tcbh_col]):
        return "Thiếu cột cần thiết!", pd.DataFrame()
    df18 = df[
        (df[product_col] == product_name) &
        (df[customer_col] == customer_name) &
        (df["Tháng"] == month)
    ]
    ok = (df18[ok_col] == 1).sum()
    fail = (df18[fail_col] == 1).sum()
    tcbh = (df18[tcbh_col] == 1).sum()
    total = ok + fail + tcbh
    percent = round(ok / total * 100, 2) if total > 0 else 0
    return f"Tỷ lệ sửa thành công {product_name} của {customer_name} trong tháng {month}", pd.DataFrame({"Tổng xử lý": [total], "Sửa thành công (%)": [percent]})

def query_19_top_technicians(df, top_n=5):
    tech_col = find_col(df.columns, "kỹ thuật viên")
    if not tech_col:
        return "Không có cột 'Kỹ thuật viên'", pd.DataFrame()
    top_ktv = df[tech_col].value_counts().head(top_n)
    return f"Top kỹ thuật viên xử lý nhiều sản phẩm nhất", pd.DataFrame({"Kỹ thuật viên": top_ktv.index, "Số lượng": top_ktv.values})

def query_20_success_rate_by_technician_and_group(df, group_by):
    tech_col = find_col(df.columns, "kỹ thuật viên")
    ok_col = find_col(df.columns, "đã sửa xong")
    fail_col = find_col(df.columns, "không sửa được")
    tcbh_col = find_col(df.columns, "từ chối bảo hành")
    if not all([tech_col, ok_col, fail_col, tcbh_col]):
        return "Thiếu cột kỹ thuật viên hoặc trạng thái!", pd.DataFrame()
    df2 = df.copy()
    df2["OK"] = (df2[ok_col] == 1).astype(int)
    df2["FAIL"] = (df2[fail_col] == 1).astype(int)
    df2["TCBH"] = (df2[tcbh_col] == 1).astype(int)
    g = df2.groupby([group_by, tech_col]).agg(
        ok=("OK", "sum"),
        fail=("FAIL", "sum"),
        tcbh=("TCBH", "sum")
    ).reset_index()
    g["Tỷ lệ sửa thành công (%)"] = round(g["ok"] / (g["ok"] + g["fail"] + g["tcbh"]) * 100, 2)
    return f"Tỷ lệ sửa thành công của kỹ thuật viên theo {group_by.lower()}", g[[group_by, tech_col, "ok", "fail", "tcbh", "Tỷ lệ sửa thành công (%)"]]

def query_21_technician_status_summary(df):
    tech_col = find_col(df.columns, "kỹ thuật viên")
    ok_col = find_col(df.columns, "đã sửa xong")
    fail_col = find_col(df.columns, "không sửa được")
    tcbh_col = find_col(df.columns, "từ chối bảo hành")
    if not all([tech_col, ok_col, fail_col, tcbh_col]):
        return "Thiếu cột kỹ thuật viên hoặc trạng thái!", pd.DataFrame()
    g = df.groupby(tech_col).agg(
        ok=(ok_col, lambda x: (x == 1).sum()),
        fail=(fail_col, lambda x: (x == 1).sum()),
        tcbh=(tcbh_col, lambda x: (x == 1).sum()),
    ).reset_index()
    g["Tổng xử lý"] = g["ok"] + g["fail"] + g["tcbh"]
    g["Tỷ lệ sửa thành công (%)"] = g.apply(
        lambda row: round(row["ok"] / row["Tổng xử lý"] * 100, 2) if row["Tổng xử lý"] > 0 else 0, axis=1
    )
    g = g.rename(columns={
        "ok": "Sửa xong",
        "fail": "Không sửa được",
        "tcbh": "Từ chối BH"
    })
    return "Thống kê số lượng sản phẩm mỗi kỹ thuật viên đã xử lý", g
