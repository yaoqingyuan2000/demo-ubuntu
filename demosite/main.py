 
from demosite.wsgi import application

from wsgiref.simple_server import make_server


if __name__ == '__main__':

    server = make_server('127.0.0.1', 8080, application)

    server.serve_forever()