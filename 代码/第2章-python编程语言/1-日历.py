# 2.1 日历的输出，通过输入年份和月份，显示出指定月份的日历
#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 判断是否是闰年
def leap_year(year):
    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        return True
    else:
        return False


# 获取每个月的天数
def month_days(year, month):
    if month == 2:
        if (leap_year(year)):
            return 29
        else:
            return 28
    elif month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    else:
        return 30


# 获取从当前年到1990年的总天数
def total_days(year, month):
    days = 0
    for i in range(1990, year):
        if (leap_year(i)):
            days += 366
        else:
            days += 365
    for i in range(1, month):
        days += month_days(year, i)
    return days


# 主程序
if __name__ == '__main__':
    year = 2018
    month = 10
    print('    {}年{}月份日历'.format(year, month))
    print('Sun    Mon    Tues    Wed    Thur    Fri    Sat')
    print('---------------------------------------------------------------')
    count = 0
    # 当前月份的1号是星期几，将前面的位置空置出来
    for i in range((total_days(year, month) + 1) % 7):
        print(end='    ')
        count += 1
    # 按星期位置输出每个月的天数
    for i in range(1, month_days(year, month) + 1):
        print(i, end='    ')

        count += 1
        if count % 7 == 0:
            print()
