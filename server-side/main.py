from wsgiref.simple_server import make_server


import time

class IPBlacklistMiddleware(object):
    def __init__(self, app):
        self.app = app
 
    def __call__(self, environ, start_response):
        ip_addr = environ.get('HTTP_HOST').split(':')[0]
        print(ip_addr)
        # if ip_addr not in ('127.0.0.1'):
        #     return forbidden(start_response)
 
        return self.app(environ, start_response)
 
def dog(start_response):
    start_response('200 OK', [('Content-Type', 'text/plain')])
    time.sleep(10)
    return ['This is dog!'.encode("utf-8")]
 
def cat(start_response):
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return ['This is cat!'.encode("utf-8")]
 
def not_found(start_response):
    start_response('404 NOT FOUND', [('Content-Type', 'text/plain')])
    return ['Not Found'.encode("utf-8")]
 
def forbidden(start_response):
    start_response('403 Forbidden', [('Content-Type', 'text/plain')])
    return ['Forbidden'.encode("utf-8")]
 
def application(environ, start_response):
    path = environ.get('PATH_INFO', '').lstrip('/')
    mapping = {'dog': dog, 'cat': cat}
 
    call_back = mapping[path] if path in mapping else not_found
    return call_back(start_response)
 
if __name__ == '__main__':

    application = IPBlacklistMiddleware(application)
    server = make_server('127.0.0.1', 8080, application)


    server.serve_forever()