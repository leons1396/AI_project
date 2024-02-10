def encoding_labels(df, col_name, values):
    df.loc[df[col_name] == 'Karotte', col_name] = values[0]  #0
    df.loc[df[col_name] == 'Kartoffel', col_name] = values[1]  #1
    df.loc[df[col_name] == 'Zwiebel', col_name] = values[2]  #2
    df.loc[df[col_name] == 'Karotte_Trieb', col_name] = values[3]  #3
    df.loc[df[col_name] == 'Kartoffel_Trieb', col_name] = values[4]  #4
    df.loc[df[col_name] == 'Zwiebel_Trieb', col_name] = values[5]  #5
    
    return df

def do():
    print("say hello")