
import pandas as pd
import unicodedata
import re

def clean_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    text = text.replace('đ', 'd').replace('Đ', 'd')
    text = text.lower().strip()
    text = re.sub(r'[\W_]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def normalize_for_match(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    text = re.sub(r'[\-_\&]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def match_block(name, keyword):
    n_name = normalize_for_match(name)
    n_key = normalize_for_match(keyword)
    n_name_words = n_name.split()
    n_key_words = n_key.split()
    for i in range(len(n_name_words) - len(n_key_words) + 1):
        if n_name_words[i:i+len(n_key_words)] == n_key_words:
            return True
    return False

def find_col(cols, keyword, column_mapping={}):
    keyword_clean = clean_text(keyword)
    mapping_found = column_mapping.get(keyword.strip(), [])

    for alias in mapping_found:
        for col in cols:
            if clean_text(col) == clean_text(alias):
                return col

    for col in cols:
        if keyword_clean == clean_text(col):
            return col
    for col in cols:
        if keyword_clean in clean_text(col):
            return col
    return None

def ensure_time_columns(df):
    date_col = None
    for col in df.columns:
        cleaned = unicodedata.normalize('NFKD', col).lower().replace('đ','d')
        cleaned = re.sub(r'[\W_]+', ' ', cleaned)
        if ("ngay" in cleaned) and (("nhan" in cleaned) or ("tiep" in cleaned)):
            date_col = col
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df["Năm"] = df[date_col].dt.year
        df["Tháng"] = df[date_col].dt.month
        df["Quý"] = df[date_col].dt.quarter
    return df

def extract_time_filter_from_question(question):
    years = re.findall(r"(20\d{2})", question)
    years = [int(y) for y in years]
    months = re.findall(r"tháng\s*(\d{1,2})", question)
    months = [int(m) for m in months if 0 < int(m) <= 12]
    quarters = re.findall(r"quý\s*([1234IiVv]+)", question)
    quarter_map = {
        "1": 1, "I": 1, "i": 1,
        "2": 2, "II": 2, "ii": 2,
        "3": 3, "III": 3, "iii": 3,
        "4": 4, "IV": 4, "iv": 4,
    }
    q_norm = []
    for q in quarters:
        qv = str(q).upper()
        if qv in quarter_map:
            q_norm.append(quarter_map[qv])
        else:
            try:
                q_norm.append(int(qv))
            except:
                pass
    quarters = [q for q in q_norm if 1 <= q <= 4]
    return years, months, quarters

def filter_df_by_time(df, years=None, months=None, quarters=None):
    df2 = df.copy()
    if years and "Năm" in df2.columns:
        df2 = df2[df2["Năm"].isin(years)]
    if months and "Tháng" in df2.columns:
        df2 = df2[df2["Tháng"].isin(months)]
    if quarters and "Quý" in df2.columns:
        df2 = df2[df2["Quý"].isin(quarters)]
    return df2
