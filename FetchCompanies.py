import sys
import time
import threading
import requests
import pandas as pd

# Global flag to stop the spinner
done_event = threading.Event()

def spinning_cursor():
    """Display a rotating spinner in the console."""
    spinner = ['|', '\\', '-', '/']
    try:
        while not done_event.is_set():  # Control loop with a condition variable
            for char in spinner:
                sys.stdout.write(f'\r{char}')
                sys.stdout.flush()
                time.sleep(0.2)
    except KeyboardInterrupt:
        sys.stdout.write('\rProcess interrupted.\n')  # Display a message on interrupt
        sys.exit(0)  # Gracefully exit the spinner thread

def fetch_company_data(alphabet):
    print(alphabet)
    """Fetch company data for a given alphabet."""
    payload = {"alphabet": alphabet}
    response = requests.post(
        "https://www.cse.lk/api/alphabetical",
        data=payload  # Use `data` to send form data
    )

    # Check for HTTP errors
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch data for {alphabet}: {response.status_code} {response.reason}")

    # Parse the JSON response
    response_data = response.json()
    return response_data

def fetch_company_details(symbol):
    """Fetch detailed company data for a given symbol."""
    payload = {"symbol": symbol}
    response = requests.post(
        "https://www.cse.lk/api/companyInfoSummery",
        data=payload  # Use `data` to send form data
    )

    # Check for HTTP errors
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch data for {symbol}: {response.status_code} {response.reason}")

    # Parse the JSON response
    response_data = response.json()
    return response_data

def fetch_data():
    all_data = []
    for letter in "A":#BCDEFGHIJKLMNOPQRSTUVWXYZ":
        try:
            data = fetch_company_data(letter)
            # Check if the response contains the key 'reqAlphabetical'
            if "reqAlphabetical" in data:
                for company in data["reqAlphabetical"]:
                    symbol = company["symbol"]
                    sys.stdout.write(f'\r{symbol}')
                    # Fetch detailed data for each company using its symbol
                    details = fetch_company_details(symbol)
                    if "reqSymbolInfo" in details:
                        company_details = details["reqSymbolInfo"]
                        # Merge company summary and detailed information
                        combined_data = {
                            "name": company_details["name"],
                            "symbol": company_details["symbol"],
                            "price": company_details["closingPrice"],
                            "percentageChange": company_details["changePercentage"],
                            "turnover": company_details["tdyTurnover"],
                            "sharevolume": company_details["tdyShareVolume"],
                            "tradevolume": company_details["tdyTradeVolume"],
                            "marketCap": company_details["marketCap"],
                            "marketCapPercentage": company_details["marketCapPercentage"],
                        }
                        all_data.append(combined_data)
            else:
                print(f"No 'reqAlphabetical' key in the response for alphabet: {letter}")       
        except Exception as e:
            print(f"Error fetching data for {letter}: {e}")
    return all_data

def rank_companies():
    """Rank companies based on specified criteria."""
    # Start the spinner in a separate thread
    spinner_thread = threading.Thread(target=spinning_cursor)
    spinner_thread.start()
    all_data = []
    try:
        all_data = fetch_data()
    finally:   
        done_event.set()  # Stop the spinner thread gracefully
        # Wait for spinner to finish
        spinner_thread.join()
        
    # Check if data was collected
    if not all_data:
        print("No data collected. Exiting.")
        return

    # Convert all_data to a DataFrame
    df = pd.DataFrame(all_data)

    # Filter for companies with turnover above 50,000, but also allow smaller companies with high growth
    df = df[df["turnover"] > 50000]  # Lowered the turnover threshold for smaller companies

    # Market Cap filter - include small but active companies
    df = df[df["marketCap"] > 500_000]  # Lower the market cap threshold to include promising small companies

    # Keep companies with significant market cap percentage
    df = df[df["marketCapPercentage"] > 0.05]  # Slightly reduce the percentage threshold for smaller players

    # Normalize and weight metrics
    df["normalized_change"] = df["percentageChange"] / df["percentageChange"].max()
    df["normalized_turnover"] = df["turnover"] / df["turnover"].max()
    df["normalized_volume"] = df["sharevolume"] / df["sharevolume"].max()
    #take a trade volume vs share volume ratio
    df["normalized_marketCap"] = df["marketCap"] / df["marketCap"].max()
    df["normalized_marketCapPercentage"] = df["marketCapPercentage"] / df["marketCapPercentage"].max()

    # Scoring weights
    weights = {
        "normalized_change": 0.4,
        "normalized_turnover": 0.2,
        "normalized_volume": 0.1,
        "normalized_marketCap": 0.2,
        "normalized_marketCapPercentage": 0.1,
    }

    # Calculate the overall score
    df["score"] = (
        df["normalized_change"] * weights["normalized_change"] +
        df["normalized_turnover"] * weights["normalized_turnover"] +
        df["normalized_volume"] * weights["normalized_volume"] +
        df["normalized_marketCap"] * weights["normalized_marketCap"] +
        df["normalized_marketCapPercentage"] * weights["normalized_marketCapPercentage"]
    )

    # Return the top 20 companies
    top_companies = df.sort_values(by="score", ascending=False).head(100)
    return top_companies[["name", "symbol", "price", "percentageChange", "turnover", "sharevolume", "tradevolume", "marketCap", "marketCapPercentage", "score"]]

