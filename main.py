import json
import openai
import pandas
import matplotlib.pyplot as plt 
import streamlit
import yfinance 

openai.api_key = open('API_KEY', 'r').read()

def getStockPrice(ticker):
    return str(yfinance.Ticker(ticker).history(period='1y').iloc[-1].Close)

def calculateEMA(ticker, window):
    data = yfinance.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])

def calculateSMA(ticker, window):
    data = yfinance.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])

def calculateRSI(ticker):
    data = yfinance.Ticker(ticker).history(period='1y').Close
    diff = data.diff()
    up = diff.clip(lower=0)
    down = -1 * diff.clip(upper=0)
    emaUp = up.ewm(com=14-1, adjust=False).mean()
    emaDown = down.ewm(com=14-1, adjust=False).mean()
    rs = emaUp / emaDown
    return str(100 -(100 / (1 + rs)).iloc[-1])

def calculateMACD(ticker):
    data = yfinance.Ticker(ticker).history(period='1y').Close
    smallEMA = data.ewm(span=12, adjust=False).mean()
    bigEMA = data.ewm(span=26, adjust=False).mean()
    MACD = smallEMA - bigEMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACDhistogram = MACD - signal 
    return f'{MACD[-1]}, {signal[-1]}, {MACDhistogram[-1]}'

def plotPrice(ticker):
    data = yfinance.Ticker(ticker).history(period='1y').Close
    plt.figure(figsize=(10,5))
    plt.plot(data.index, data.Close)
    plt.title('{ticker} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()
    
functions = [
    {
            'name': 'getStockPrice',
            'description': 'Gets latest stock price of company given ticker symbol of company.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'Stock ticker symbol for a company.'
                    }
                },
                'required': ['ticker']
        }
    },
    {
        "name": "calculateSMA",
        "description": "Calculate the simple moving average for a given stock ticker and a window.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The ticker symbol for the company.",
                },
                "window": {
                    "type": "integer",
                    "description": "The timeframe when calculating the SMA."
                }
            },
            "required": ["ticker", "window"],
        },
    },
    {
        "name": "calculateEMA",
        "description": "Calculates the exponential moving average using the provided parameters.",
        "parameters": {
            "type" : "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker symbol for the company.",
                },
                "window": {
                    "type": "integer",
                    "description": "Timeframe when calculating the EMA."
                }
            },
            "required": ["ticker", "window"],
        },
    },
    {
        "name": "calculateRSI",
        "description": "Calculates the relative strength index (RSI) of a given stock.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The ticker symbol for the company.",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculateMACD",
        "description": "Calculates the Moving Average Convergence Divergence (MACD) of given stock.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The ticker symbol for the company.",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plotPrice",
        "description": "Plots the closing price of a specific stock over time.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The ticker symbol for the company.",
                },
            },
            "required": ["ticker"],
        },
    },
]
    
functionsAvailable = {
    'getStockPrice': getStockPrice,
    'calculateSMA': calculateSMA,
    'calculateEMA': calculateEMA,
    'calculateRSI': calculateRSI,
    'calculateMACD': calculateMACD,
    'plotPrice': plotPrice
}

if 'messages' not in streamlit.session_state:
    streamlit.session_state['messages'] = []
    
streamlit.title("Stock Analyst")

userInput = streamlit.text_input('Your question:')
if userInput:
    try:
        streamlit.session_state['messages'].append({'role': 'user', 'content': f'{userInput}'})
        response = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo-0613',
            messages = streamlit.session_state['messages'],
            functions = functions,
            function_call = 'auto'
        )
        
        responseMessage = response['choices'][0]['message']
        
        if responseMessage.get('function-call'):
            funcName = responseMessage['function-call']['name']
            args = json.loads(responseMessage['function_call']['arguments'])
            if funcName in ['getStockPrice', 'calculateRSI', 'calculateMACD', 'plotPrice']:
                argsDict = {'ticker': args.get('ticker')}
            elif funcName in ['calculateSMA', 'calculateEMA']:
                argsDict = {'ticker': args.get('ticker'), 'window': args.get('window')}
                
            functionCall = functionsAvailable[funcName]
            functionResponse = functionCall(**argsDict)
            
            if funcName == 'plotPrice':
                streamlit.image('stock.png')
            else: 
                streamlit.session_state['messages'].append(responseMessage)
                streamlit.session_state['messages'].append(
                    {
                        'role': 'function',
                        'name': funcName,
                        'content': functionResponse
                    }
                )
                secondResponse = openai.ChatCompletion(
                    model='gpt-3.5-turbo-0613',
                    messages = streamlit.session_state['messages']
                )
                streamlit.text(secondResponse['choices'][0]['message']['content'])
                streamlit.session_state['messages'].append({'role': 'assistant', 'content': secondResponse['choices'][0]['message']['content']})
        else:
            streamlit.text(responseMessage['content'])
            streamlit.session_state['messages'].append({'role': 'assistant', 'content': responseMessage['content']})
    except Exception as e:
        raise e
    
