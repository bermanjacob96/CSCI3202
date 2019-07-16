def add(a,b):
    if (isinstance(a, int) and isinstance(b, int)):
        x = a + b
    elif (isinstance(a, int) and isinstance(b, float)):
        x = a + b
    elif (isinstance(a, float) and isinstance(b, int)):
        x = a + b
    elif (isinstance(a, float) and isinstance(b, float)):
        x = a + b
    elif type(a) is list and type(b) is list:
        x = a + b
    elif ((type(a) is int or type(a) is float) and type(b) is list):
        a = [a]
        x = a + b
    elif ((type(a) is int or type(a) is float) and type(b) is str):
        a = str(a)
        x = a + b
    elif ((type(b) is int or type(b) is float) and type(a) is list):
        b = [b]
        x = a + b
    elif ((type(b) is int or type(b) is float) and type(a) is str):
        b = str(b)
        x = a + b
    else:
         print("Error!")
    return x

def calcMyGrade(a_S,m_S,p_S,w):
    a_v = (sum(a_S)/100)
    m_v = (sum(m_S)/100)
    p_v = (sum(p_S)/100)
    a_w = a_v*w[0]
    m_w = m_v*w[1]
    p_w = p_v*w[2]
    w_avg = ((a_w+m_w+p_w)/3)*10
    return w_avg
