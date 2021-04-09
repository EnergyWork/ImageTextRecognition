from python3_anticaptcha import NoCaptchaTaskProxyless
# Enter the key to the AntiCaptcha service from your account. Anticaptcha service key.
ANTICAPTCHA_KEY = "39aeef98e48d826f3e6393a6e689ef5c"
# G-ReCaptcha ключ сайта. Website google key.
SITE_KEY = "6LfBoFcUAAAAAHhf5gDvAGboIPEywad4gJAdbrzx"
# Page url.
PAGE_URL = "https://www.fonts-online.ru/font/Sunday-Best-%5Brus-by-aLiNcE%5D"
# Get string for solve captcha, and other info.
user_answer = NoCaptchaTaskProxyless.NoCaptchaTaskProxyless(anticaptcha_key = ANTICAPTCHA_KEY).captcha_handler(websiteURL=PAGE_URL, websiteKey=SITE_KEY)

print(user_answer)