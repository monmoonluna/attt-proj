from flask import Flask, Response, request, redirect

app = Flask(__name__)

@app.route('/', defaults={'path': ''}, methods=['GET', 'HEAD', 'POST'])
@app.route('/<path:path>', methods=['GET', 'HEAD', 'POST'])
def catch(path):
    print("\n========== REQUEST ==========")
    print("Method:", request.method)
    print("URL:", request.url)
    print("Host:", request.headers.get("Host"))
    print("Origin:", request.headers.get("Origin"))
    print("Headers:", dict(request.headers))

    if request.method == 'HEAD':
        resp = Response("")
        resp.headers['Content-Type'] = 'text/x-component'
        return resp

    return redirect('http://backend/flag.txt')


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=35162,
        debug=True
    )