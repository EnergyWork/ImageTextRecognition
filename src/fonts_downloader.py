import os
from time import sleep

import selenium.common.exceptions as selexcept
from selenium.webdriver import Chrome, ChromeOptions

def download_font(driver, url):
    driver.get(url)
    xpath_download_button = '//*[@id="content"]/a[1]'
    download_button = driver.find_element_by_xpath(xpath_download_button)
    download_button.click()
    sleep(2)
    while True:
        files = os.listdir(download_dir)
        if not any(x.endswith('.crdownload') for x in files):
            break
        else:
            print('Идёт скачивание...')
    print('Файл успешно скачен')

webdriver = "C:\\Programs\\chromdriver\\chromedriver.exe"

download_dir = 'd:\\StudyMagistracy\\CourseProject\\data\\dataset3\\fonts\\'
prefs = {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade" : True,
    "download.extensions_to_open" : ""
}
browser_options = ChromeOptions()
browser_options.headless = True
browser_options.add_argument("--disable-gpu")
browser_options.add_argument("--no-sandbox")
browser_options.add_argument("disable-infobars")
browser_options.add_argument("--disable-extensions")
browser_options.add_experimental_option('prefs', prefs)

driver = Chrome(executable_path=webdriver, options=browser_options)

urls = [
    "https://fonts.mega8.ru/index.php?font=4284&name=a_StamperRg%26amp%3BBt%20Regular",
    "https://fonts.mega8.ru/index.php?font=3415&name=Century%20Gothic%20Regular",
    "https://fonts.mega8.ru/index.php?font=4271&name=AG_Cooper%20Italic",
    "https://fonts.mega8.ru/index.php?font=236&name=Inform%20Bold",
    "https://fonts.mega8.ru/index.php?font=6155&name=PF%20Fuel%20Pro%20Regular",
    "https://fonts.mega8.ru/index.php?font=4982&name=EUROSTYLE_CYR%20Regular",
    "https://fonts.mega8.ru/index.php?font=4372&name=China%20Regular"
]
for url in urls:
    download_font(driver, url)

driver.close()
