#DouBanBookScrapy.py
# 导入相应的库文件
from lxml import etree
import requests
import csv

# 创建csv
fp = open('C:/Users/56943/PycharmProjects/pythonProject/result.csv', 'wt', newline='', encoding='utf-8')
writer = csv.writer(fp)
writer.writerow(('name', 'url', 'author', 'publisher', 'date', 'price', 'date', 'price', 'rate', 'comment'))

# 构造url
urls = ['https://book.douban.com/top250?start={}'.format(str(i)) for i in range(0, 250, 25)]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'
}

for url in urls:
    html = requests.get(url, headers=headers)
    selector = etree.HTML(html.text)
    infos = selector.xpath('//tr[@class="item"]')
    for info in infos:
        name = info.xpath('td/div/a/@title')[0]
        url = info.xpath('td/div/a/@href')[0]
        book_infos = info.xpath('td/p/text()')[0]
        author = book_infos.split('/')[0]
        publisher = book_infos.split('/')[-3]
        date = book_infos.split('/')[-2]
        price = book_infos.split('/')[-1]
        rate = info.xpath('td/div/span[2]/text()')[0]
        comments = info.xpath('td/p/span/text()')
        comment = comments[0] if len(comments) != 0 else "空"
        writer.writerow((name, url, author, publisher, date, price, date, price, rate, comment))

fp.close()

#DouBanExcelData.py
#导入需要的包
import os
import pandas as pd
import xlsxwriter

#创建和打开Excel文件路径
if 'myXlsxFolder' not in os.listdir():
    os.mkdir('myXlsxFolder')
os.chdir('myXlsxFolder')

#将result.csv的数据按需求导入指定的Excel表格
#输入爬取的数据
books_data = pd.read_csv('../result.csv',usecols=['name','author','publisher','price','rate','comment','url'],na_values = 'NULL')
titles = books_data['name']
authors = books_data['author']
publishers = books_data['publisher']
prices = books_data['price']
ratings = books_data['rate']
comments = books_data['comment']
urls = books_data['url']
#创建电子表格文件book_info.xlsx，并为其添加一个名为“豆瓣图书”的工作表
myXlsxFile = xlsxwriter.Workbook('book_info.xlsx')
myWorkSheet = myXlsxFile.add_worksheet('豆瓣图书')

nums = len(titles)  #根据标题数量获取记录数

#第一行写入列名
myWorkSheet.write(0,0,'图书标题')
myWorkSheet.write(0,1,'图书作者')
myWorkSheet.write(0,2,'出版社')
myWorkSheet.write(0,3,'图书价格')
myWorkSheet.write(0,4,'图书评分')
myWorkSheet.write(0,5,'图书简介')
myWorkSheet.write(0,6,'资源地址')
#设置列宽
myWorkSheet.set_column('A:A',20)
myWorkSheet.set_column('B:B',20)
myWorkSheet.set_column('C:C',30)
myWorkSheet.set_column('D:D',10)
myWorkSheet.set_column('E:E',10)
myWorkSheet.set_column('F:F',50)
myWorkSheet.set_column('G:G',30)
#写入图书数据
for i in range(1,nums):
    myWorkSheet.write(i, 0, titles[i])
    myWorkSheet.write(i, 1, authors[i])
    myWorkSheet.write(i, 2, publishers[i])
    myWorkSheet.write(i, 3, prices[i])
    myWorkSheet.write(i, 4, ratings[i])
    myWorkSheet.write(i, 5, comments[i])
    myWorkSheet.write(i, 6, urls[i])

myXlsxFile.close()  #关闭电子表格