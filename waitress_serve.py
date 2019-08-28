
# -*- coding: utf-8 -*-
"""Run a waitress server to run the Flask application

Example:

        $ python waitress_serve.py

"""
from flasker import app,FluServer

from waitress import serve

if __name__ == "__main__":
        serv = FluServer()
        serve(app, host="0.0.0.0", port=9000)