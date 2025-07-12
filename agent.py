from io import BytesIO
from time import sleep

import helium
from dotenv import load_dotenv
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from smolagents import CodeAgent, tool
from smolagents.agents import ActionStep

# Load environment variables
load_dotenv()

@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.

    Args:
        text (str): The text to be searched for
        nth_result (int): Which occurence to jump to (default: 1)
    """
    elements = webdriver.find_elements(By.XPATH, f"//*[contains(text()), '{text}']")
    if nth_result > len(elements):
        raise Exception(f"Match number {nth_result} not found (only {len(elements)} matches found)")

    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    webdriver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result

@tool
def go_back() -> None:
    """Goes back to the previous page"""
    webdriver.back()
    
@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows!
    But this will not work on cookie consent banners.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()