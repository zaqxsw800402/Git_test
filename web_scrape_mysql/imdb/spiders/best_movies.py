# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from ..items import ImdbItem


class BestMoviesSpider(CrawlSpider):
    name = 'best_movies'
    allowed_domains = ['web.archive.org']

    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36'

    def start_requests(self):
        yield scrapy.Request(url='http://web.archive.org/web/20200715000935if_/https://www.imdb.com/search/title/?groups=top_250&sort=user_rating', headers={
            'User-Agent': self.user_agent
        })

    rules = (
        Rule(LinkExtractor(restrict_xpaths="//h3[@class='lister-item-header']/a"), callback='parse_item', follow=True, process_request='set_user_agent'),
        Rule(LinkExtractor(restrict_xpaths="(//a[@class='lister-page-next next-page'])[2]"), process_request='set_user_agent')
    )

    def set_user_agent(self, request, spider):
        request.headers['User-Agent'] = self.user_agent
        return request

    def parse_item(self, response):
        items = ImdbItem()
        items['title'] = response.xpath("//div[@class='title_wrapper']/h1/text()").get()
        items['year'] = response.xpath("//span[@id='titleYear']/a/text()").get()
        items['duration'] = response.xpath("normalize-space((//time)[1]/text())").get()
        items['genre']= response.xpath("//div[@class='subtext']/a[1]/text()").get()
        items['rating']= response.xpath("//span[@itemprop='ratingValue']/text()").get()
        items['movie_url']=response.url
        # yield {
        #     'title': response.xpath("//div[@class='title_wrapper']/h1/text()").get(),
        #     'year': response.xpath("//span[@id='titleYear']/a/text()").get(),
        #     'duration': response.xpath("normalize-space((//time)[1]/text())").get(),
        #     'genre': response.xpath("//div[@class='subtext']/a[1]/text()").get(),
        #     'rating': response.xpath("//span[@itemprop='ratingValue']/text()").get(),
        #     'movie_url': response.url
        # }
        yield {
            'title': items['title'],
            'year': items['year'],
            'duration': items['duration'],
            'genre': items['genre'],
            'rating': items['rating'],
            'movie_url':  items['movie_url']
        }
        # yield {items}
