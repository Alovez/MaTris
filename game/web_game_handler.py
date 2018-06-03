from selenium import webdriver
from time import sleep
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
# enable browser logging
d = DesiredCapabilities.CHROME
d['loggingPrefs'] = { 'browser':'ALL' }

chrome_options = Options()
chrome_options.binary_location = '/opt/google/chrome/chrome'

driver = webdriver.Chrome(chrome_options=chrome_options, desired_capabilities=d)

driver.get('http://alovez.cc/Tetris')

sleep(10)
a = driver.get_log('browser')

canvas = driver.find_element_by_css_selector('#GameCanvas')

canvas.send_keys('w')

print(driver.get_log('browser'))
print(a)

canvas.send_keys('w')

print(driver.get_log('browser'))

p = input('Next Operation: ')

while p != 'q':
    canvas.send_keys(p)
    print(driver.get_log('browser'))
    p = input('Next Operation: ')

driver.quit()