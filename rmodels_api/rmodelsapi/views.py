from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework import status
import json 

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np 

# activate the pandas to R conversion
pandas2ri.activate()


@api_view(['POST'])
def predict(request):
    try:
        #load r libraries 
        robjects.r('library(base)')
        readRDS = robjects.r['readRDS']
        #read rds file
        model = readRDS('model_logit_aprovacao.rds') #if this doesn't work add the full path to the file

        data = json.loads(request.body)
        
        data_list = data.get('data_list')

        if data_list is None:
            return JsonResponse({'error': 'data_list is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        df = pd.DataFrame(data_list, columns=['tentativas_1', 'tentativas_3', 'tentativas_7', 'sucessos_1',
       'sucessos_2', 'sucessos_3', 'sucessos_6', 'entrega_3', 'entrega_6', 'entrega_7'])
        
        #calls r function 'predict' 
        prediction = predict_aprovacao(df, model)[0,1]

        #convert the r object returned from prediction back to a python object
        response_data = {'prediction': prediction} #json that is going to be returned
        return JsonResponse(response_data, status=status.HTTP_200_OK)
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    

def predict_aprovacao(df, model_logit):
    # newx = robjects.r['as.matrix'](pandas2ri.py2rpy(df))
    pred = robjects.r.predict(model_logit, newdata=df, type="response")
    pred_array = np.array(pred)
    complement_array = 1 - pred_array
    result = np.column_stack((complement_array, pred_array))
    return result