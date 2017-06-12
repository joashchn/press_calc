# -- coding: utf-8 --
# import wav_cnn
import img_cnn
import wav_cnn
import press_cnn
import os   #Python的标准库中的os模块包含普遍的操作系统功能
import re   #引入正则表达式对象
import urllib   #用于对URL进行编解码
import cgi
# import cgitb; cgitb.enable()
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler  #导入HTTP处理相关的模块

#自定义处理程序，用于处理HTTP请求
class TestHTTPHandler(BaseHTTPRequestHandler):
    #处理GET请求
    def do_GET(self):
        # if self.path == '/':
        #     self.path = '\\test.html'
        #
        # try:
        #     reply = False
        #     if self.path.endswith('.html'):
        #         reply = True
        #         mimeType = 'text/html'
        #
        #     if self.path.endswith('.jpg'):
        #         reply = True
        #         mimeType = 'image.jpg'
        #
        #     if self.path.endswith('.js'):
        #         reply = True
        #         mimeType = 'application/javascript'
        #
        #     if self.path.endswith('.txt'):
        #         reply = True
        #         mimeType = 'text/txt'
        #
        #     if (reply == True):
        #         fp = open(curdir + sep + self.path)
        #         self.send_response(200)
        #         self.send_header('content-type', mimeType)
        #         self.end_headers()
        #         self.wfile.write(fp.read())
        #         fp.close()
        #     return
        # except IOError:
        #     self.send_error(404, 'Not Found File %s' % self.path)

        # 将正则表达式编译成Pattern对象
        # pattern = re.compile(r'/qr\?s=([^\&]+)\&qr=Show\+QR')
        # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
        # match = pattern.match(self.path)

        if self.path == "/wav":
            #页面输出模板字符串
            index_content = ''
            file=open("data/wav.htm","r")
            index_content+=file.read()
            file.close()

            self.protocal_version = 'HTTP/1.1'  #设置协议版本
            self.send_response(200) #设置响应状态码
            self.send_header("Welcome", "Contect")  #设置响应头
            self.end_headers()
            self.wfile.write(index_content)   #输出响应内容
            return

        if self.path == "/press":
            #页面输出模板字符串
            index_content = ''
            file=open("data/press.htm","r")
            index_content+=file.read()
            file.close()

            self.protocal_version = 'HTTP/1.1'  #设置协议版本
            self.send_response(200) #设置响应状态码
            self.send_header("Welcome", "Contect")  #设置响应头
            self.end_headers()
            self.wfile.write(index_content)   #输出响应内容
            return

        if self.path == "/img":
            #页面输出模板字符串
            index_content = ''
            file=open("data/img.htm","r")
            index_content+=file.read()
            file.close()

            # 将正则表达式编译成Pattern对象
            # pattern = re.compile(r'/qr\?s=([^\&]+)\&qr=Show\+QR')
            # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
            # match = pattern.match(self.path)

            self.protocal_version = 'HTTP/1.1'  #设置协议版本
            self.send_response(200) #设置响应状态码
            self.send_header("Welcome", "Contect")  #设置响应头
            self.end_headers()
            self.wfile.write(index_content)   #输出响应内容
            return

    def do_POST(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type'],
                     }
        )
        # form = cgi.FieldStorage(self)

        if self.path == "/wav":
            self.send_response(200)
            self.end_headers()
            self.wfile.write("<html>")
            # self.wfile.write('Client: %sn ' % str(self.client_address) )
            # self.wfile.write('User-agent: %sn' % str(self.headers['user-agent']))
            # self.wfile.write('Path: %sn'%self.path)
            # self.wfile.write('Form data:n')
            # for field in form.keys():
            #     field_item = form[field]
            #     filename = field_item.filename
            #     filevalue  = field_item.value
            #     filesize = len(filevalue)#文件大小(字节)
            #     print len(filevalue)
            #     with open(filename.decode('utf-8')+'a','wb') as f:
            #         f.write(filevalue)


            # mpath,margs=urllib.splitquery(self.path)
            # datas = self.rfile.read(int(self.headers['content-length']))
            #
            # result = cnn.test(datas[2:])
            #
            # self.protocal_version = 'HTTP/1.1'  #设置协议版本
            # self.send_response(200) #设置响应状态码
            # self.send_header("Welcome", "Contect")  #设置响应头
            # self.end_headers()
            result = wav_cnn.test(form['s'].value)
            self.wfile.write(result)   #输出响应内容
            self.wfile.write("</html>")
            return

        if self.path == "/press":
            self.send_response(200)
            self.end_headers()
            self.wfile.write("<html>")
            result = press_cnn.test(form['s'].value)
            self.wfile.write(result)  # 输出响应内容

            self.wfile.write("</html>")
            return

        if self.path == "/img":
            self.send_response(200)
            self.end_headers()
            self.wfile.write("<html>")

            fileitem = form['f']

            if fileitem.filename:
                tname = "test.png"
                img = fileitem.file.read()
                with open(tname,'wb') as f:
                    f.write(img)
                result = img_cnn.test(img)
                self.wfile.write(result)   #输出响应内容
            self.wfile.write("</html>")
            return

#启动服务函数
def start_server(port):
    try:
        http_server = HTTPServer(('', int(port)), TestHTTPHandler)
        print '\n\nStart HTTP Server at PORT:' , port
        http_server.serve_forever() #设置一直监听并接收请求
    except KeyboardInterrupt:
        print 'Shutting down the server!!'
        http_server.socket.close()
if __name__=='__main__':
    # os.chdir('/tmp')  #改变工作目录到 static 目录
    start_server(8000)  #启动服务，监听8000端口
