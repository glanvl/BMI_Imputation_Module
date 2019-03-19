import Algorithmia
import pandas as pd
from io import StringIO
import json
# API calls will begin at the apply() method, with the request body passed as 'input'
# For more details, see algorithmia.com/developers/algorithm-development/languages

def apply(input):
    finalResult = getFile()
    #finalResult = finalResult.to_csv(index=False)
    print("HERE?")
    Algorithmia.client().file("data://glanvl/BMIData/output_Lamb_Test.csv").put(str(finalResult))
    return finalResult #.to_json(orient='records')[1:-1].replace('}, {','},{')
    
def getFile():

    client = Algorithmia.client()
   # path = "data://kingmike/testcollection/mike2.csv"
    file = client.file("data://glanvl/BMIData/luke2.csv")
    period = 201809
    dataframe = pd.read_csv(StringIO(file.getString()),index_col=False)
    # kingmike/BMIImputationCheck4NonResp
    responderdata = checknonresp(dataframe,period)
    #responderdata = callCheckNonResp(dataframe)
    print(str(responderdata))
    movementData = doMovement(responderdata,period)
    moves = add_reg_fourteen(movementData)
    movesdata= pd.DataFrame(moves)
    means = calculateMeans(movesdata,moves)
    iqrs = get_iqrs(means, movesdata)
    
        
    input_data = pd.merge(iqrs, movesdata, how='left', on=['region', 'strata', 'responder_id'])
    non_atyp_data=input_data.apply(remAtypicals,axis=1)
    print(str(non_atyp_data.columns.values))
    remeaned_data = recalc_means(non_atyp_data).drop_duplicates()
    gbimp_data= remeaned_data[(remeaned_data.region == 14)].apply(calcGBImputation,axis=1)
    print(str(gbimp_data.columns.values))
    #reorder and drop unneeded rows
    gbimp_data = gbimp_data[['strata','gbimp1','gbimp2','gbimp3','gbimp4','gbimp5','gbimp6','gbimp7']]
    output_data = remeaned_data.merge(gbimp_data,left_on = ['strata'],right_on = ['strata'],how='inner')
    #filter out any data without a region or from a diff period
    impstat=filterData(output_data,period)
    impstat=impstat.apply(assignMoM,axis= 1) 
      
    datasetandimps = prep_for_appimp(dataframe,impstat, period)
    imputeddataset = datasetandimps.apply(applyFactors,axis=1)
  #      
    callIQR(dataframe['Q601_asphalting_sand'])
        
    #return "mike"  

    return imputeddataset.to_json(orient='records')

def get_iqrs(means, moves):
   print('Calculating interquartile ranges....')   
   
   iqrs = pd.DataFrame()
   
   # Loops through and filters the moves dataset so that each individual
   # region and strata groupping is processed individually. 
   for row in means.values:
       iqr_filter = (moves["region"] == row[2]) & (moves["strata"] == row[3])
       filtered_iqr = moves[iqr_filter]
       # Pass the question number and region and strata groupping to the 
       # iqr_sum function.
       iqr601 = iqr_sum(filtered_iqr,'movement_Q601_asphalting_sand')
       iqr602 = iqr_sum(filtered_iqr,'movement_Q602_building_soft_sand')
       iqr603 = iqr_sum(filtered_iqr,'movement_Q603_concreting_sand')
       iqr604 = iqr_sum(filtered_iqr,'movement_Q604_bituminous_gravel')
       iqr605 = iqr_sum(filtered_iqr,'movement_Q605_concreting_gravel')
       iqr606 = iqr_sum(filtered_iqr,'movement_Q606_other_gravel')
       iqr607 = iqr_sum(filtered_iqr,'movement_Q607_constructional_fill')
       # Append the variables to the temporary iqrs dataset
       iqrs =  iqrs.append({'region': row[2], 'strata': row[3], 
                            'iqr_601': iqr601, 'iqr_602': iqr602,
                            'iqr_603': iqr603, 'iqr_604': iqr604, 
                            'iqr_605': iqr605, 'iqr_606': iqr606,
                            'iqr_607': iqr607}, ignore_index=True)
   # Merge the temporaary dataset with the iqrs to the means dataset to create 
   # the final output.
   
   iqr_final = pd.merge(means, iqrs.drop_duplicates(), how='left', on=['region', 'strata']) 
   return iqr_final
   
def iqr_sum(df,quest):
    '''
    Inputs: 
    df - Working dataset with the month on month question value movements 
    filtered by each individual combination of region and strata.
    
    quest - Individual question no
    
    Returns:
    Returns the iqr for the question value based on the region, strata
    and question number being passed through.
    '''
    
    #df=dfTest
    df=df[quest]
    
    #df=pd.Series(df)
    dfSize=df.size
    import math
    #df=df.apply(unneg)
    if(dfSize%2 == 0):

       sortedDF = df.sort_values()
       df=sortedDF.reset_index(drop=True)
       dfbottom=df[0:math.ceil(int(dfSize/2))].median() 
       dftop=df[math.ceil(int(dfSize/2)):].median()
       iqr = dftop-dfbottom
       #iqr = quantile75 - quantile25
    else:
       sortedDF = df.sort_values()
       df=sortedDF.reset_index(drop=True)
       q1 = df[(math.ceil(0.25*(dfSize+1)))-1]
       q3 = df[(math.floor(0.75*(dfSize+1)))-1]
       iqr = q3 - q1
    
    return iqr 


    
def checknonresp(data,period):
    print('Checking for non response....')
    
    print(str(data))
    #Create a dataframe where the response column value is set as 1 i.e non responders
    non_response_filter = (data["response_type"] == 1) & (data["period"] == period)
    filtered_non_responders = data[non_response_filter]
    # check the length of the dataset - if there are rows then run imputation...
    response_check = len(filtered_non_responders.index)
    if response_check > 0:
        print('Non Responders identified, calculating imputation factors...')
        # Ensure that only responder_ids with a response type of 2 (returned) get
        # picked up
        data=data[data['response_type']==2]
        # Select and order the columns required for imputation.
        ordered_data = data[['responder_id','land_or_marine','region','strata','Q601_asphalting_sand',
                           'Q602_building_soft_sand','Q603_concreting_sand','Q604_bituminous_gravel','Q605_concreting_gravel','Q606_other_gravel',
                           'Q607_constructional_fill','Q608_total','period'

        ]]
        return ordered_data
    else:
        print('There are no non responders!')
        print('Skipping Imputation!')
        return

def add_reg_fourteen(unagg_data):
               # Take a copy of the input dataset
    reg14data = unagg_data.copy()
               # Set the region value to 14 (GB)
    reg14data['region'] = 14
               # Append the new reg14 dataset to the input dataset
    try:
        full_dataset = unagg_data.append(reg14data)
    except AttributeError as AError:
        full_dataset=unagg_data
    return full_dataset




def doMovement(input_data,period):

    prev = calculatePrevPeriod(period)
    print('Calculating period on period percentage movement....')
               
    # Split the input data set into the previous and current period and then 
    # merge these datasets side by side so that derived columns can be calculated.
    prev = input_data[input_data.period == prev ]
               #prev = input_data[period == prev]
    current = input_data[input_data.period == period]
    merged_df = prev.merge(current, left_on = ['responder_id','land_or_marine','region'],
                                right_on = ['responder_id','land_or_marine','region'],
                                how='inner')            
    
    # Fix Strata issues. Ensure that only only one STRATA value exists for each responder_id
               # in the merged dataframe. 

    merged_df = merged_df.drop('strata_x', axis=1)
    merged_df.rename(columns={'strata_y':'strata'},inplace=True)

    
    
               # Calculate the % movement value for each question, based on the question values for the
    # current and previous periods.             The apply function calls the MovementCalc method.
    #return merged_df
    merged_df['movement_Q601_asphalting_sand'] = merged_df.apply(lambda x: MovementCalc(x[3], x[13]),axis=1)
    merged_df['movement_Q602_building_soft_sand'] = merged_df.apply(lambda x: MovementCalc(x[4], x[14]),axis=1)
    merged_df['movement_Q603_concreting_sand'] = merged_df.apply(lambda x: MovementCalc(x[5], x[15]),axis=1)
    merged_df['movement_Q604_bituminous_gravel'] = merged_df.apply(lambda x: MovementCalc(x[6], x[16]),axis=1) 
    merged_df['movement_Q605_concreting_gravel'] = merged_df.apply(lambda x: MovementCalc(x[7], x[17]),axis=1)
    merged_df['movement_Q606_other_gravel'] = merged_df.apply(lambda x: MovementCalc(x[8], x[18]),axis=1)
    merged_df['movement_Q607_constructional_fill'] = merged_df.apply(lambda x: MovementCalc(x[9], x[19]),axis=1)   
    
               # Drop unneeded columns.
    
    merged_df.drop(['Q601_asphalting_sand_x','Q602_building_soft_sand_x','Q603_concreting_sand_x','Q604_bituminous_gravel_x','Q605_concreting_gravel_x',
                'Q606_other_gravel_x','Q607_constructional_fill_x','Q608_total_x','period_x','period_y'],axis =1, inplace = True)
       
               # Rename required columns 
               
    merged_df.rename(columns={'Q601_asphalting_sand_y': 'Q601_asphalting_sand',
                       'Q602_building_soft_sand_y': 'Q602_building_soft_sand',
                       'Q603_concreting_sand_y': 'Q603_concreting_sand',
                       'Q604_bituminous_gravel_y': 'Q604_bituminous_gravel',
                       'Q605_concreting_gravel_y': 'Q605_concreting_gravel',
                       'Q606_other_gravel_y': 'Q606_other_gravel',
                       'Q607_constructional_fill_y': 'Q607_constructional_fill',
                       'Q608_total_y': 'Q608_total'
                       }, inplace = True)
    
    return merged_df
    
    
def calculatePrevPeriod(period):
    strperiod = str(period)
    month = strperiod[4:6]
    year = strperiod[0:4]
    
    if month == '03':
        month = int(month)+9
        year = int(year)-1
        month = str(month)
        year = str(year)
    else:
        month = int(month)-3
        month = '0' + str(month)
        month = str(month)
    
    prevperiod = year + month
    prevperiod = int(prevperiod)
    return prevperiod
  
def callCheckNonResp(dataframe):
    client = Algorithmia.client()
    algo = client.algo('kingmike/BMIImputationCheck4NonResp').set_options(stdout=True)
    try:
        # Get the summary result of your file's contents
        print(dataframe.to_json(orient='records'))
        print(algo.pipe(dataframe.to_json(orient='records')).result)
    except Exception as error:
        # Algorithm error if, for example, the input is not correctly formatted
        print("bad")
        print(error)
        
def callIQR(series):

    client = Algorithmia.client()
    algo = client.algo('kingmike/pythonIQR')
    print(type(series))
    print(str(json.dumps(series.tolist())))
    # Pass in input required by algorithm
    try:
        # Get the summary result of your file's contents
        print(algo.pipe(series.tolist()).result)
    except Exception as error:
        # Algorithm error if, for example, the input is not correctly formatted
        print("bad")
        print(error)
        
        
def MovementCalc(previous,current):
    '''
               Input:
               previous - Question value in the previous period
               current - Question value in the current period.
              
               Returns:
               Movement - A percentage movement value based on the two values passed through
               to this method.
               '''
    if int (previous) > 0: Movement = ((int (current) - int (previous))/ int (previous))
    else: Movement = 0
    return Movement   

def calculateMeans(dataframe1,dataframe2):
   print('Calculating mean question value movement....')
    
   #Dropping columns not needed for the calculation
   dataframe1.drop(['land_or_marine','Q601_asphalting_sand','Q602_building_soft_sand',
               'Q603_concreting_sand','Q604_bituminous_gravel','Q605_concreting_gravel','Q606_other_gravel',
               'Q607_constructional_fill','Q608_total'],axis = 1, inplace = True)
    
   #Create dataframe which counts the movements grouped by region and strata
   counts = dataframe1.groupby(['region','strata']).count()
   #Rename columns to fit naming standards
   counts.rename(columns={'movement_Q601_asphalting_sand': 'movement_Q601_asphalting_sand_count',
                           'movement_Q602_building_soft_sand': 'movement_Q602_building_soft_sand_count',
                           'movement_Q603_concreting_sand': 'movement_Q603_concreting_sand_count',
                           'movement_Q604_bituminous_gravel': 'movement_Q604_bituminous_gravel_count',
                           'movement_Q605_concreting_gravel': 'movement_Q605_concreting_gravel_count',
                           'movement_Q606_other_gravel': 'movement_Q606_other_gravel_count',
                           'movement_Q607_constructional_fill': 'movement_Q607_constructional_fill_count'}
                           , inplace = True)
                         
   #Create dataframe which sums the movements grouped by region and strata                    
   sums = dataframe1.groupby(['region','strata']).sum()
   #Rename columns to fit naming standards
   sums.rename(columns={'movement_Q601_asphalting_sand': 'movement_Q601_asphalting_sand_aggregated',
                         'movement_Q602_building_soft_sand': 'movement_Q602_building_soft_sand_aggregated',
                         'movement_Q603_concreting_sand': 'movement_Q603_concreting_sand_aggregated',
                         'movement_Q604_bituminous_gravel': 'movement_Q604_bituminous_gravel_aggregated',
                         'movement_Q605_concreting_gravel': 'movement_Q605_concreting_gravel_aggregated',
                         'movement_Q606_other_gravel': 'movement_Q606_other_gravel_aggregated',
                         'movement_Q607_constructional_fill': 'movement_Q607_constructional_fill_aggregated'}, inplace = True)
                         
   counts1 = counts.reset_index(level = ['region','strata'])
   sums1 = sums.reset_index(level = ['region','strata'])
   moves1 = sums1.merge(counts1,left_on = ['region','strata'],
                         right_on = ['region','strata'], how = 'left')
                         
   moves1['mean_601'] = moves1.apply(lambda x: meanMove(x,3),axis= 1)
   moves1['mean_602'] = moves1.apply(lambda x: meanMove(x,4),axis= 1)
   moves1['mean_603'] = moves1.apply(lambda x: meanMove(x,5),axis= 1)
   moves1['mean_604'] = moves1.apply(lambda x: meanMove(x,6),axis= 1)
   moves1['mean_605'] = moves1.apply(lambda x: meanMove(x,7),axis= 1)
   moves1['mean_606'] = moves1.apply(lambda x: meanMove(x,8),axis= 1)
   moves1['mean_607'] = moves1.apply(lambda x: meanMove(x,9),axis= 1)
    
   dataframe2.drop(['movement_Q601_asphalting_sand','movement_Q602_building_soft_sand','movement_Q603_concreting_sand',
               'movement_Q604_bituminous_gravel','movement_Q605_concreting_gravel','movement_Q606_other_gravel',
               'movement_Q607_constructional_fill','Q601_asphalting_sand','Q602_building_soft_sand','Q603_concreting_sand',
               'Q604_bituminous_gravel','Q605_concreting_gravel','Q606_other_gravel','Q607_constructional_fill',
               'Q608_total'],axis =1, inplace = True)
               
   means = dataframe2.merge(moves1,left_on = ['region','strata'],
                       right_on = ['region','strata'], how = 'left')
                                                                              
   moves1.drop(['responder_id_x','responder_id_y'],axis =1, inplace = True)
   
   means = dataframe2.merge(moves1,left_on = ['region','strata'],
                    right_on = ['region','strata'], how = 'left')
   return means
   
def meanMove(dataframe,col_no):
   '''
   Inputs:
   Moves - Non aggregated dataframe with the percentage month on month
   movement for each of the questions.
   Col_no - Location of value within column number specified
   
   Returns:
   Means - Dataframe with mean month on month question value movements
   aggregated by Region and Strata.
   '''
   #If the value at the location plus 7 (Count) is greater than 0
   if int(dataframe[col_no+7]) >0:
   # Calculate the mean 
       meanMovement = dataframe[col_no]/dataframe[col_no+7]
   #If value = 0 then the mean movement = 0
   else: meanMovement = 0
   return meanMovement

def remAtypicals(row):

    '''
    Removes atypical movements from the dataset.
    As part of the process, will also add 7 atyp columns
    
    Method is designed to be used with an apply() method: In which every row of a dataset 
    is passed through and processed.    
    
    Params: row - A row of the dataset
            
    Returns: row - The row of the dataset once processed

               '''
    '''
    Original method was designed for differnt column names:
    q601_value, q602_value, q603_value etc
    so it was able to use a loop to hit each of the columns
      That is no longer the case, so we do a similar thing by looping through a list of column names
    '''
    
    def docalculation(row,qval,x):
        
        row['atypical_'+str(qval)]=abs(row['movement_'+str(qval)]-row['mean_60'+str(x)])-2*row['iqr_60'+str(x)]  
        if(row['atypical_'+str(qval)]>0):
            #Movement becomes nan (Count method ignores NaNs)
            row['movement_'+str(qval)]=None
    
        return row
    
    questions= ['Q601_asphalting_sand','Q602_building_soft_sand','Q603_concreting_sand',
                'Q604_bituminous_gravel','Q605_concreting_gravel','Q606_other_gravel','Q607_constructional_fill']
    x=1            
    for question in questions:
        row=docalculation(row,question,x)
        x+=1
    
        
    
    return row 

import numpy as np
def meanMove(dataframe,col_no):
   '''
   Inputs:
   Moves - Non aggregated dataframe with the percentage month on month
   movement for each of the questions.
   Col_no - Location of value within column number specified
   
   Returns:
   Means - Dataframe with mean month on month question value movements
   aggregated by Region and Strata.
   '''
   #Altered so's to not allow any Nans into the calculation
   #If the value at the location plus 7 (Count) is greater than 0     
   if(np.isnan(dataframe[col_no]) ==False):
       if int(dataframe[col_no+7]) >0:
           # Calculate the mean 
           meanMovement = dataframe[col_no]/dataframe[col_no+7]
   #If value = 0 then the mean movement = 0
       else: meanMovement = 0
   else: meanMovement = 0
   return meanMovement
   
def recalc_means(working_data):
     print('Recalculating Means....')
     working_data = working_data[['region','strata','land_or_marine','movement_Q601_asphalting_sand',
                           'movement_Q602_building_soft_sand','movement_Q603_concreting_sand','movement_Q604_bituminous_gravel',
                           'movement_Q605_concreting_gravel','movement_Q606_other_gravel','movement_Q607_constructional_fill'
     ]]
     
     #Altered to create a lormdata dataset to add on later(this could be improved by getting lorm from strata)
     lormdata=working_data[['region','strata','land_or_marine']]
     
     
     #Create dataframe which counts the movements grouped by region and strata
     #Altered to specify which rows we are counting
     counts = working_data.groupby(['region','strata'])[['movement_Q601_asphalting_sand','movement_Q602_building_soft_sand','movement_Q603_concreting_sand','movement_Q604_bituminous_gravel','movement_Q605_concreting_gravel','movement_Q606_other_gravel','movement_Q607_constructional_fill']].count()
     #Index reset to facilitate merge(i think)
     counts = counts.reset_index(level = ['region','strata']) 
     #Merge lormdata on so we have lorm(see above, this could be improved)
     counts=counts.merge(lormdata,left_on = ['region','strata'],
                         right_on = ['region','strata'], how = 'left')  
     
     
     #Rename columns to fit naming standards
     counts.rename(columns={'movement_Q601_asphalting_sand': 'movement_Q601_asphalting_sand_count',
                           'movement_Q602_building_soft_sand': 'movement_Q602_building_soft_sand_count',
                           'movement_Q603_concreting_sand': 'movement_Q603_concreting_sand_count',
                           'movement_Q604_bituminous_gravel': 'movement_Q604_bituminous_gravel_count',
                           'movement_Q605_concreting_gravel': 'movement_Q605_concreting_gravel_count',
                           'movement_Q606_other_gravel': 'movement_Q606_other_gravel_count',
                           'movement_Q607_constructional_fill': 'movement_Q607_constructional_fill_count'}
                           , inplace = True)
                           
      
     #Create dataframe which sums the movements grouped by region and strata                    
     sums = working_data.groupby(['region','strata']).sum()
     #Rename columns to fit naming standards
     sums.rename(columns={'movement_Q601_asphalting_sand': 'movement_Q601_asphalting_sand_aggregated',
                         'movement_Q602_building_soft_sand': 'movement_Q602_building_soft_sand_aggregated',
                         'movement_Q603_concreting_sand': 'movement_Q603_concreting_sand_aggregated',
                         'movement_Q604_bituminous_gravel': 'movement_Q604_bituminous_gravel_aggregated',
                         'movement_Q605_concreting_gravel': 'movement_Q605_concreting_gravel_aggregated',
                         'movement_Q606_other_gravel': 'movement_Q606_other_gravel_aggregated',
                         'movement_Q607_constructional_fill': 'movement_Q607_constructional_fill_aggregated'}, inplace = True)
                         
     counts1 = counts.reset_index(level = ['region','strata'])
     sums1 = sums.reset_index(level = ['region','strata'])
     new_moves = sums1.merge(counts1,left_on = ['region','strata'],
                         right_on = ['region','strata'], how = 'left')
     #left with a wayward 'index' column. Drop it.
     new_moves.drop(['index'],axis =1, inplace = True)
                        
     new_moves['mean_601'] = new_moves.apply(lambda x: meanMove(x,2),axis= 1)
     new_moves['mean_602'] = new_moves.apply(lambda x: meanMove(x,3),axis= 1)
     new_moves['mean_603'] = new_moves.apply(lambda x: meanMove(x,4),axis= 1)
     new_moves['mean_604'] = new_moves.apply(lambda x: meanMove(x,5),axis= 1)
     new_moves['mean_605'] = new_moves.apply(lambda x: meanMove(x,6),axis= 1)
     new_moves['mean_606'] = new_moves.apply(lambda x: meanMove(x,7),axis= 1)
     new_moves['mean_607'] = new_moves.apply(lambda x: meanMove(x,8),axis= 1) 
     
     return new_moves

def calcGBImputation(row):
    
    def docalculation(row,qval,x):
        print("Current x is: "+str(x))
        if(row['movement_'+str(qval)+'_count']<3):
            #check for l or m, set gbimp accordingly
            if(row['land_or_marine']=='L'):
                row['gbimp'+str(x)] = 0
            elif(row['land_or_marine']=='M'):
                row['gbimp'+str(x)] = 1
        else:
            #otherwise, set to mean of movement
            row['gbimp'+str(x)] = row['mean_60'+str(x)]
        return row
            
    questions= ['Q601_asphalting_sand','Q602_building_soft_sand','Q603_concreting_sand',
                'Q604_bituminous_gravel','Q605_concreting_gravel','Q606_other_gravel','Q607_constructional_fill']
        
      
    x=1
    #For each question name in the list. Pass the row, question name, and a counter to an inner function to perform the calculation
    for question in questions:
        row=docalculation(row,question,x)
        x+=1
    print(str(row))                
    return row
def filterData(impstat,period):
    #impstat['region']=impstat['region'].astype(str)#
    import numpy as np
    print('Removing NaNs....')
    
    #Filter set so's not to include any nans
    impstat=impstat[impstat['region']!=np.nan]
    #add on period
    impstat['period'] = period
    return impstat

def assignMoM(row):
    
    def docalculation(row,qval,x):
        if(row['movement_'+str(qval)+'_count']<5):
            #imp factor becomes that of region 14
            row['Imp'+str(x)]=row['gbimp'+str(x)]
        else:
            #otherwise, if mean is nan set factor to 0
            if(str(row['mean_60'+str(x)]) =='NaN' ):
                row['Imp'+str(x)]=0
            #if mean isn't nan, set it as the imputation factor
            else:
                row['Imp'+str(x)]=row['mean_60'+str(x)]
        row['Imp'+str(x)]=row['Imp'+str(x)]+1
        return row        
    questions= ['Q601_asphalting_sand','Q602_building_soft_sand','Q603_concreting_sand',
                'Q604_bituminous_gravel','Q605_concreting_gravel','Q606_other_gravel','Q607_constructional_fill']
    x=1            
    #For each question name in the list. Pass the row, question name, and a counter to an inner function to perform the calculation
  
    for question in questions:
        row=docalculation(row,question,x)
        x+=1
    
    
    return row


def prep_for_appimp(dataset,impfactors,period):
    import pandas as pd
    #Subset of impfactors
    impfactors   =  impfactors[['region','strata','period','Imp1','Imp2','Imp3','Imp4','Imp5','Imp6','Imp7']]
    #mergeddataset will contain previous and current values for each non responder
    #We fill in current values, so perhaps not necessary to include them as part of mergetdataset
    mergeddataset =  gather_non_responders(dataset,period)
    #merge the factors and values so they are all on one row
    datasetandimps = pd.merge(mergeddataset,impfactors,how='inner',on=['region','strata'])
    #Both sets had period, so we drop one
    datasetandimps.drop(['period_x'],axis =1, inplace = True)
    datasetandimps.rename(columns={'period_y': 'period'}, inplace = True)
    print('Applying factors')
    return datasetandimps

def applyFactors(row):
    #For each column, perform the sum. 
    #  Value = (PreviousValue*ImputationFactor)+0   (rounded to nearest whole number)
    # +0 is a redundant step so has been removed
    
   # Note: Sas does not impute for column 8(total), rather summing the other columns.
   #   However the number should  be the same as if it were imputed
   #   (if not, then that would suggest problems with imputation)
   '''
   Original method was designed for differnt column names:
   q601_value, q602_value, q603_value etc
   so it was able to use a loop to hit each of the columns
   That is no longer the case, so we do a similar thing by looping through a list of column names
   For each question name in the list. Pass the row, question name, and a counter to an inner function to perform the sum
   '''
   def docalculation(row,qval,x):
       row[str(qval)]=round(row['Previous_'+str(qval)]*row['Imp'+str(x)],0)
       return row
        
   questions= ['Q601_asphalting_sand','Q602_building_soft_sand','Q603_concreting_sand',
                'Q604_bituminous_gravel','Q605_concreting_gravel','Q606_other_gravel','Q607_constructional_fill']
   x=1             
   for question in questions:
       row=docalculation(row,question,x) 
       x+=1       
        
   row['Q608_total'] = sum((row['Q601_asphalting_sand'],row['Q602_building_soft_sand'],row['Q603_concreting_sand'],row['Q604_bituminous_gravel'],row['Q605_concreting_gravel'],row['Q606_other_gravel'],row['Q607_constructional_fill']))
   return row

def save_imputed_values(imputeddataset,disagg,period):
    #drop unneeded values now that imputation has been applied(may not be necessary in new world)
    imputeddataset.drop(['Previous_Q601_asphalting_sand','Previous_Q602_building_soft_sand','Previous_Q603_concreting_sand','Previous_Q604_bituminous_gravel','Previous_Q605_concreting_gravel','Previous_Q606_other_gravel','Previous_Q607_constructional_fill','Previous_Q608_total'
                        ,'Imp1','Imp2','Imp3','Imp4','Imp5','Imp6','Imp7'],axis =1, inplace = True)
    #Remove current period from disagg
    #imputedrus=imputeddataset['responder_id']
    #disaggNotCurrent = disagg.query('(period != 201706) | ((period == 201706 & response_type==2))')
    #nonresponders = disagg[disagg['responder_id'].isin(imputeddataset['responder_id'])]
    
    
    disaggNotCurrent = disagg[~((disagg.period == period) & (disagg['responder_id'].isin(imputeddataset['responder_id'])))]
    #append imputed data to disagg
    disaggComplete=disaggNotCurrent.append(imputeddataset)
    return disaggComplete
    
def gather_non_responders(data,period):
    import pandas as pd
    prev = calculatePrevPeriod(period)
    #Create a dataframe where the response column value is set as 1 i.e non responders
    non_filter = (data["response_type"] == 1) & (data["period"] == period)
    CurrentNonResponse = data[non_filter]
    pnon_filter = (data["period"]==prev)
    Previous = data[pnon_filter]
    
    Previous.rename(columns={'Q601_asphalting_sand': 'Previous_Q601_asphalting_sand',
                           'Q602_building_soft_sand': 'Previous_Q602_building_soft_sand',
                            'Q603_concreting_sand': 'Previous_Q603_concreting_sand',
                             'Q604_bituminous_gravel': 'Previous_Q604_bituminous_gravel',
                            'Q605_concreting_gravel': 'Previous_Q605_concreting_gravel',
                             'Q606_other_gravel': 'Previous_Q606_other_gravel',
                            'Q607_constructional_fill': 'Previous_Q607_constructional_fill',
                            'Q608_total': 'Previous_Q608_total'
        }, inplace = True)    
        
    
    Previous.drop(['period'], axis = 1, inplace = True)
    
    disaggMerged = pd.merge(CurrentNonResponse,Previous,how='left',
                            on=['responder_id','land_or_marine','gor_code',
                            'county','region_name','county_name','enterprise_ref'
                            ])

    #Rename columns
    disaggMerged.rename(columns={'Q601_asphalting_sand_y': 'Previous_Q601_asphalting_sand',
                       'Q602_building_soft_sand_y': 'Previous_Q602_building_soft_sand',
                      'Q603_concreting_sand_y': 'Previous_Q603_concreting_sand',
                       'Q604_bituminous_gravel_y': 'Previous_Q604_bituminous_gravel',
                       'Q605_concreting_gravel_y': 'Previous_Q605_concreting_gravel',
                       'Q606_other_gravel_y': 'Previous_Q606_other_gravel',
                       'Q607_constructional_fill_y': 'Previous_Q607_constructional_fill',
                       'Q608_total_y': 'Previous_Q608_total',
                       'Q601_asphalting_sand_x': 'Q601_asphalting_sand',
                       'Q602_building_soft_sand_x': 'Q602_building_soft_sand',
                       'Q603_concreting_sand_x': 'Q603_concreting_sand',
                       'Q604_bituminous_gravel_x': 'Q604_bituminous_gravel',
                       'Q605_concreting_gravel_x': 'Q605_concreting_gravel',
                       'Q606_other_gravel_x': 'Q606_other_gravel',
                       'Q607_constructional_fill_x': 'Q607_constructional_fill',
                       'Q608_total_x': 'Q608_total',
                       'region_x'    : 'region',
                       'strata_x'    : 'strata',
                       'response_type_x':'response_type',
                       'name_x':'name',
                       'period_x':'period',
                       'land_or_marine_y':'land_or_marine'
                       }, inplace = True)
              

    disaggMerged.drop(['region_y','response_type_y','strata_y'],axis =1, inplace = True)
     
    return disaggMerged
