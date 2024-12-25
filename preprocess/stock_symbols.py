import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

class StockSymbols:
    def __init__(self):
        self.data = {}

        html = self._fetch_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        table_data = self._parse_table(html)
        self._process_symbols(table_data)
        self.filter(['ANET', 'HUBB'])

    def __len__(self):
        '''Return the number of stock symbols'''
        return len(self.data)
    
    def __getitem__(self, symbol: str):
        '''Return the sector and industry for the given symbol'''
        return self.data[symbol]
    
    def __getitem__(self, index: int):
        '''Return the symbol at the given index'''
        return list(self.data.keys())[index]
    
    def __iter__(self):
        '''Return an iterator for the stock symbols'''
        return iter(self.data.keys())
    
    def audit(self, incl_symbols):
        '''Update the symbol data to only include the given symbols'''
        self.data = {symbol: self.data[symbol] for symbol in incl_symbols}

    def filter(self, rem_symbols):
        '''Filter out the symbols that are in the given list'''
        self.data = {symbol: self.data[symbol] for symbol in self.data if symbol not in rem_symbols}

    def _fetch_html(self, url):
        '''Fetch HTML content from the given URL'''
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def _parse_table(self, html):
        '''Parse the HTML to extract the table data'''
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        return pd.read_html(StringIO(str(table)))[0].to_dict('records')

    def _process_symbols(self, table_data):
        '''Process the table data to extract symbols, sectors, and industries'''
        sectors = set()
        industries = set()

        for company in table_data:
            symbol = company['Symbol'].replace('.', '-')

            sectors.add(company['GICS Sector'])
            sector = list(sectors).index(company['GICS Sector']) + 1

            industries.add(company['GICS Sub-Industry'])
            industry = list(industries).index(company['GICS Sub-Industry']) + 1

            self.data[symbol] = (sector, industry)
