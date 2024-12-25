from bs4 import BeautifulSoup
import requests

def scrape_cse():
    # Replace this URL with the official CSE market page
    url = "https://www.cse.lk/pages/market-summary/market-summary.component.html"

    try:
        response = requests.get(url)
        print("respond", response.status_code)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            print(soup)
            # Example: Scrape specific stock data (replace the selectors based on the site's HTML structure)
            stocks = soup.find_all("div", class_="stock-row")  # Adjust based on the structure
            for stock in stocks:
                name = stock.find("span", class_="stock-name").text
                price = stock.find("span", class_="stock-price").text
                change = stock.find("span", class_="stock-change").text
                print(f"{name}: {price} ({change})")
        else:
            print(f"Failed to retrieve data. HTTP Status Code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    scrape_cse()
