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
from smolagents import InferenceClientModel

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
    elements = driver.find_elements(By.XPATH, f"//*[contains(text()), '{text}']")
    if nth_result > len(elements):
        raise Exception(f"Match number {nth_result} not found (only {len(elements)} matches found)")

    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result

@tool
def go_back() -> None:
    """Goes back to the previous page"""
    driver.back()
    
@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows!
    But this will not work on cookie consent banners.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
    
# Configure Chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--force-device-scale-factor=1")
chrome_options.add_argument("--window-size=1000,1350")
chrome_options.add_argument("--disable-pdf-viewer")
chrome_options.add_argument("--window-position=0,0")

driver = helium.start_chrome(headless=False, options=chrome_options)

def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)
    driver = helium.get_driver()
    if driver is not None:
        for previous_memory_step in agent.memory.steps:
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                previous_memory_step.observations_images = None
            png_bytes = driver.get_screenshot_as_png()
            image = Image.open(BytesIO(png_bytes))
            print(f"Captured a browser screenshot: {image.size} pixels")
            memory_step.observations_images = [image.copy()]
            
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )

model_id = "Qwen/Qwen2-VL-72B-Instruct"
model = InferenceClientModel(model_id=model_id)

agent = CodeAgent(
    tools = [go_back, close_popups, search_item_ctrl_f],
    model = model,
    additional_authorized_imports= ["helium"],
    step_callbacks = [save_screenshot],
    max_steps = 20,
    verbosity_level = 2,
)

agent.python_executor("from helium import *", agent.state)