import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import os
import sys

pat = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Inputs.py'
sys.path.append(pat)
app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
path = '/Bitcoin-Predictor'
os.chdir(path)
    # Layout of Dash App HTML

app.layout = html.Div(
                children=[html.Div(
                            html.Button('Detect', id='button'),
                            html.Div(id='output-container-button',
                            children='Hit the button to update.')
                    ),
                ],
            )

@app.callback(
    [Output('output-container-button', 'children')],
    [Input('button', 'n_clicks')])

script_fn = '
exec(open(script_fn).read())

''' 
def run_script_onClick(n_clicks):
    # Don't run unless the button has been pressed...
    if not n_clicks:
        raise PreventUpdate

    script_path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Inputs.py'
    # The output of a script is always done through a file dump.
    # Let's just say this call dumps some data into an `output_file`
    call(["python3", script_path])

    # Load your output file with "some code"
    output_content = lastodash
    # Now return.
    return output_content
# Main
'''
if __name__ == "__main__":
    app.run_server()

