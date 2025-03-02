import os
from flask import Flask, render_template_string
from SVM_app import server as dash_server
import json

# Add route for README to the Dash server
@dash_server.route('/')
def serve_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    template = '''
    <!DOCTYPE html>
    <html>
        <head>
            <title>SVM Interactive Demo</title>
            <meta charset="utf-8">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/sindresorhus/github-markdown-css@4.0.0/github-markdown.css">
            <script src="https://cdn.jsdelivr.net/npm/marked@4.0.0/marked.min.js"></script>
            <script>
                MathJax = {
                    tex: {
                        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                        processEscapes: true
                    }
                };
            </script>
            <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
            <style>
                .markdown-body {
                    box-sizing: border-box;
                    min-width: 200px;
                    max-width: 980px;
                    margin: 0 auto;
                    padding: 45px;
                }
                @media (max-width: 767px) {
                    .markdown-body {
                        padding: 15px;
                    }
                }
                .markdown-body pre {
                    background-color: #f6f8fa;
                    border-radius: 6px;
                    padding: 16px;
                }
                .markdown-body code {
                    background-color: rgba(175,184,193,0.2);
                    border-radius: 6px;
                    padding: 0.2em 0.4em;
                    font-size: 85%;
                }
            </style>
        </head>
        <body class="markdown-body">
            <div id="content"></div>
            <script>
                marked.setOptions({
                    gfm: true,
                    breaks: true,
                    pedantic: false,
                    smartLists: true,
                    smartypants: true,
                    xhtml: true
                });
                
                var content = ''' + json.dumps(content) + ''';
                document.getElementById('content').innerHTML = marked.parse(content);
                
                // After markdown is rendered, tell MathJax to find and render math
                if (typeof MathJax !== 'undefined') {
                    MathJax.typesetPromise();
                }
            </script>
        </body>
    </html>
    '''
    return render_template_string(template)

if __name__ == '__main__':
    if 'SPACE_ID' in os.environ:
        port = int(os.environ.get('PORT', 7860))
        dash_server.run(host='0.0.0.0', port=port)
    else:
        dash_server.run(debug=True, port=8050)