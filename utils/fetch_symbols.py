import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

def get_sp500_symbols():
    '''Get the stock symbols'''
    symbols = []
    sector = {}
    industry = {}

    sectors = set()
    industries = set()

    # Fetch the list of S&P 500 companies from Wikipedia
    page = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable', 'id': 'constituents'})
    table = StringIO(str(table))
    constituents = pd.read_html(table)[0].to_dict('records')

    # Extract the stock symbols, sectors, and industries
    for company in constituents:
        company['Symbol'] = company['Symbol'].replace('.', '-')
        symbols.append(company['Symbol'])
        sectors.add(company['GICS Sector'])
        industries.add(company['GICS Sub-Industry'])
        sector[company['Symbol']] = list(sectors).index(company['GICS Sector'])+1
        industry[company['Symbol']] = list(industries).index(company['GICS Sub-Industry'])+1

    # Remove problematic symbols
    problem_symbols = ['ANET', 'HUBB']
    symbols = [symbol for symbol in symbols if symbol not in problem_symbols]
    
    return symbols, sector, industry
