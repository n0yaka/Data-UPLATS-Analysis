def count_attendance(read_df):
    
    read_df['ATTENDANCE'] = (read_df[['M1','M2','M3']] == '/').sum(axis=1) >= 2

    return read_df