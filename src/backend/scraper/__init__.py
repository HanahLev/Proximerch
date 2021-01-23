from bs4 import BeautifulSoup
import requests


class PageScraper:
    def __init__(self, address):
        self.address = address

    def findObjective(self, objective):
        r = requests.get(self.address)
        soup = BeautifulSoup(r.content, 'html.parser')
        for element in soup.body.findAll(text=objective):
            print(element)


p = PageScraper("https://www.cambridgecheese.com/")
p.findObjective("Cheese")
