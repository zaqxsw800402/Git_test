# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from ..items import ImdbItem


class BestMoviesSpider(CrawlSpider):
    name = 'new_movies'
    allowed_domains = ['www.imdb.com']

    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36'

    def start_requests(self):
        yield scrapy.Request(url='https://www.imdb.com/chart/top/?ref_=nv_mv_250', headers={
            'User-Agent': self.user_agent
        })

    rules = (
        Rule(LinkExtractor(restrict_xpaths="//td[@class='titleColumn']/a"), callback='parse_item', follow=True,
             process_request='set_user_agent'),
        # Rule(LinkExtractor(restrict_xpaths="(//a[@class='lister-page-next next-page'])[2]"), process_request='set_user_agent')
    )

    def set_user_agent(self, request, spider):
        request.headers['User-Agent'] = self.user_agent
        return request

    def parse_item(self, response):
        items = ImdbItem()
        items['title'] = response.xpath("//h1[@class='TitleHeader__TitleText-sc-1wu6n3d-0 dxSWFG']/text()").get()
        items['year'] = response.xpath(
            "//span[@class='TitleBlockMetaData__ListItemText-sc-12ein40-2 jedhex']/text()").get()
        items['duration'] = response.xpath("//li[@class='ipc-inline-list__item'][3]/text()").get()
        items['genre'] = response.xpath("//span[@class='ipc-chip__text']/text()").get()
        items['rating'] = response.xpath(
            "//span[@class='AggregateRatingButton__RatingScore-sc-1ll29m0-1 iTLWoV']/text()").get()
        items['movie_url'] = response.url
        yield {
            'title': items['title'],
            'year': items['year'],
            'duration': items['duration'],
            'genre': items['genre'],
            'rating': items['rating'],
            'movie_url': items['movie_url']
        }
        # yield {items}
