import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import base64


# external JavaScript files
external_scripts = [
    'https://www.google-analytics.com/analytics.js',
    {'src': 'https://cdn.polyfill.io/v2/polyfill.min.js'},
    {
        'src': 'https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.10/lodash.core.js',
        'integrity': 'sha256-Qqd/EfdABZUcAxjOkMi8eGEivtdTkh3b65xCZL4qAQA=',
        'crossorigin': 'anonymous'
    }
]

# external CSS stylesheets
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }
]

app = dash.Dash(__name__,
                external_scripts=external_scripts,
                external_stylesheets=external_stylesheets)

list_of_ourgraphs = [
        '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_BM.png',
        '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_DNN.png',
        '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_DTR.png',
        '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_EN.png',
        '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_GBR.png',
        '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_KNR.png',
        '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_Lasso.png',
        '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_LR.png',
        '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_RFR.png',
        '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_Sequential.png',
        '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_SVM.png'
]




app.layout = html.Div([
    html.H1('hello',className = 'col-5'),

    dcc.Markdown('''

Inline code snippet: `True`

Block code snippet:
```py
import dash

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
```
''',
                 style={'text-align': 'center', 'display':'flex', 'vertical-align': 'middle'}),



    dcc.Dropdown(id='graph_dropdown_BTC_price',
                    options=[{'label': i, 'value': i} for i in list_of_ourgraphs],
                        value=list_of_ourgraphs[0],
                            placeholder="Select a method"
                ),

    html.Img(id='BTC_image',
                style={'text-align': 'center', 'display':'flex', 'width': '100%',
                    'max-width': '800px','vertical-align': 'middle'},
                    className = 'col-2'
             )
        ]
)

@app.callback(
    Output(component_id='BTC_image', component_property='src'),
    [Input(component_id='graph_dropdown_BTC_price',component_property= 'value')])
def update_image_src(image_path):
    print('current image_path = {}'.format(image_path))
    encoded_image = base64.b64encode(open(image_path, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())

if __name__ == '__main__':
    app.run_server(debug=True)


'''

       dcc.Markdown(tx.BM_txt_markdown,
                    style={'display': 'inline-block', 'vertical-align': 'middle'
                           }
                    ),



@app.callback(
    [Output(component_id='text-display',component_property='children')],
    [Input(component_id='text-input',component_property='value')])

def update_text_output_2(input_value):
    with open(path1, 'r') as Texlist1:
        content = Texlist1.read()
    return content
'''
