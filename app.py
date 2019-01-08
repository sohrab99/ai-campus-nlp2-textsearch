#!/usr/bin/env python3
import os
import sys
import hashlib
from newspaper import Article   # https://github.com/codelucas/newspaper
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main ():
    url = request.values.get('url', None)
    if url is None:
        return render_template('index.html', article=None)
    # process url
    # sample URL: https://www.cnbc.com/2019/01/08/chairman-eddie-lampert-to-get-another-chance-to-save-sears-sources-say.html
    print(url)
    ar = Article(url)
    ar.download()
    ar.parse()
    # ar.nlp()   Not working yet
    # ar.text contains the text
    # see more in https://github.com/codelucas/newspaper
    return render_template('index.html',
                            article=ar)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777, debug=True)

