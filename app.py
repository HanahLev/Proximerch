from algo import *
from key import *
import flask

app = flask.Flask(__name__, template_folder='templates')

#Set up the main route
@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
            
    if flask.request.method == 'POST':
        s_name = flask.request.form['search_name']
        postcode = flask.request.form['loc_name']
        source_id = get_placeid(postcode)
        final_res_dict, final_sorted_res = order_shops_final(source_id, s_name, top_places[:20])
        
        if final_res_dict is None:
            #return(flask.render_template('error.html'))
            pass
        else:
            return flask.render_template('results2.html',name = s_name, recommended=final_sorted_res)

if __name__ == '__main__':
    app.run()
