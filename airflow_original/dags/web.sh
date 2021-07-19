#!/bin/bash

cd /opt/webscrape/bookstoscrape
echo SPIDER START
scrapy crawl books -o new.csv