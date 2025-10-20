import os 
import dotenv
import json
from typing import List, Optional

from pydantic import BaseModel, Field
import asyncio
from flask import Flask, render_template, request, redirect, url_for, flash
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

dotenv.load_dotenv('.env.local')
model = ChatOpenAI(model_name="gpt-4o")

server_params = StdioServerParameters(
    command='npx',
    args=['@brightdata/mcp'],
    env = {
        'API_TOKEN': os.getenv('API_TOKEN'),
        'BROWSER_AUTH' : os.getenv('BROWSER_AUTH'),
        'WEB_UNLOCKER_ZONE': os.getenv('WEB_UNLOCKER_ZONE')
    }
)

SYSTEM_PROMPT = """You are an AI agent designed to assist users with e-commerce related tasks. 
                    You have access to a variety of tools to help you gather information, analyze data, 
                    and provide recommendations. Always strive to understand the user's needs and provide 
                    accurate, helpful responses."""

PLATFORMS = ['Amazon', 'eBay', 'Noon']

class Hit(BaseModel):
    title: str = Field(..., description="Title of the product")
    url: str = Field(..., description="URL link to the product")
    rating: str = Field(..., description="Rating of the product")

class PlatformBlock(BaseModel):
    platform: str = Field(..., description="E-commerce platform name")
    hits: List[Hit] = Field(..., description="List of top product hits")

class ProductSearchResponse(BaseModel):
    platforms: List[PlatformBlock] = Field(..., description="Search results from various platforms")


app = Flask(__name__)
app.secret_key = os.urandom(24)

async def run_agent(query: str, platformms):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()

            tools = await load_mcp_tools(sess)

            agent = create_agent(model=model, tools=tools, response_format=ProductSearchResponse)

            # sanitize prompt later
            prompt= f'{query}\n\n Search on the following platforms: {", ".join(platformms)}'

            result = await agent.ainvoke({
                'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                             {'role': 'user', 'content': prompt}]
            })

            structured_response = result['structured_response']

            return structured_response.model_dump()


@app.route('/', methods=['GET', 'POST'])
def index():
    query = ""
    platforms = []
    response_json = None 
    if request.method == 'POST':
        query = request.form.get("query", "").strip()
        platforms = request.form.getlist("platforms")
        if not query:
            flash("Please enter a search query.", "danger")
            return redirect(url_for('index'))
        if not platforms:
            flash("Please select at least one platform.", "danger")
            return redirect(url_for('index'))

    
        try:
            response_json = asyncio.run(run_agent(query, platforms))
        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f"Agent error: {e}", "danger")
            return redirect(url_for('index'))

    return render_template(
            "index.html",
            query=query,
            platforms=PLATFORMS,
            selected=platforms,
            response=response_json,
        )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)