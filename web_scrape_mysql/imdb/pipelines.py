# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import logging
import MySQLdb
from scrapy.exceptions import DropItem

class DuplicatesTitlePipeline(object):
    def __init__(self):
        self.article = set()
    def process_item(self, item, spider):
        title = item['title']
        if title in self.article:
            raise DropItem('duplicates title found %s', item)
        self.article.add(title)
        return(item)

class DeleteNullTitlePipeline(object):
    def process_item(self, item, spider):
        title = item['title']
        if title:
            return item
        else:
            raise DropItem('found null title %s', item)

class MysqlPipeline(object):
    collection_name = "best_movies"

    # def __init__(self, host, user, passwd, db):
    def __init__(self, host, user, passwd, db, use_unicode=True, charset='utf-8'):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            host=crawler.settings.get('HOST'),
            user=crawler.settings.get('USER'),
            passwd=crawler.settings.get('PASSWD'),
            db=crawler.settings.get('DB')
        )

    def open_spider(self, spider):
        self.db = MySQLdb.connect(self.host, self.user, self.passwd, self.db, charset='utf8')
        self.cursor = self.db.cursor()
        try:
            self.cursor.execute('''
                CREATE TABLE new_movies_pro(
                    title TEXT,
                    year TEXT,
                    duration TEXT,
                    genre TEXT,
                    rating FLOAT,
                    movie_url TEXT)
            ''')
            self.db.commit()
        except MySQLdb.OperationalError:
            pass

    def process_item(self, item, spider):
        item['rating'] = float(item['rating'])
        try:

            self.cursor.execute("""
                INSERT INTO new_movies_pro (title, year, duration, genre, rating, movie_url) 
                VALUES(%s,%s,%s,%s,%s,%s)""", (
                # item.get('title', 'Not Available').replace("\xa0", ''),
                item.get('title').encode('utf-8'),
                item.get('year'),
                item.get('duration'),
                item.get('genre'),
                item.get('rating'),
                item.get('movie_url'),
            ))
            self.db.commit()

        except MySQLdb.Error as e:
            logging.error(e)

        return item

    def close_spdier(self, item, spider):
        self.db.close()

