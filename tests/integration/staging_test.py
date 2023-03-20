# Test a ML service by sending a JSON payload to a scoring URL and checking the response. Before deploying into to production
import requests # to make http requests
import json # to encode and decode JSON data

#create test data sample
test_sample = json.dumps({'data': [[0.0,50.81694915254237,17.396155965078933,24.0,38.0,50.0,61.5,82.0,1.8125,0.3559322033898305,0.4576271186440678,0.0238095238095238,0.2301587301587301,0.0396825396825396,0.246031746031746,0.0714285714285714,0.2222222222222222,0.0873015873015873,0.0476190476190476,0.0317460317460317,0.0,40186.56716417911,50192.28708540663,15300.0,24000.0,33500.0,43750.0,67699.99999999999,1982.8,1998.0,2002.0,2006.0,2008.0,2082.733333333333,989.9377728068824,1149.0,1596.0,1961.0,2320.0,3512.499999999997,2.029850746268657,1.2427992506377683,1.0,1.0,2.0,2.5,5.0,3296.4186567164184,5276.69979614379,200.0,911.8,1698.4,3499.85,11341.739999999994,645.6871260582208,514.64,915.5,1257.2,1652.05,2428.4,16.0,1,295,1824,21.6,63.2,15.2],[0.003,51.4,18.1,24.2,37.4,50.8,64.4,81.9,1.71,0.368,0.420,0.028,0.20,0.033,0.14,0.138,0.23,0.171,0.018,0.052,0.003,35067,18000,16516,23366,31838,41758,64000,1997,2003,2008,2011,2014,1813,593,1141,1428,1739,2011,2826,2.35,1.63,1.01,1.07,1.95,3.03,5.3,3748,5000,288,1003,2188,4518,11000,620,628,914,1210,1555,2381,10,4,300,1000,21.5,40.6,30]]})
test_sample = str(test_sample)


def test_ml_service(scoreurl, scorekey):
    # check if the the scoreurl parameter is not None
    assert scoreurl != None

    # if the scorekey is provided, authenticate the request
    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + scorekey)}

    # http post request
    resp = requests.post(scoreurl, test_sample, headers=headers)
    assert resp.status_code == requests.codes.ok # check if the status code is 200 OK
    assert resp.text != None
    assert resp.headers.get('content-type') == 'application/json'
    assert int(resp.headers.get('Content-Length')) > 0

