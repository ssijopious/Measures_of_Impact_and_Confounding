# Categorical to numerical Coercion

def ca_coe(dfInput,infVariable,trgtVariable):

    for dtColumn in dfInput :

        # Checking for the categorical columns
        if dfInput[dtColumn].dtypes == 'O' and dtColumn in infVariable: 

            dfavg=(
                dfInput.groupby(dtColumn)[trgtVariable].mean()
            ).sort_values(trgtVariable) 

            # Avg of target column values according to the distince
            # values of influvencing column and sort according to it
            # (hot number encoding with groupby)
            dfavg['row_num'] = np.arange(len(dfavg)) 

            # Removing the hierarchical index of dataframe
            dfavg.reset_index(inplace=True)      

            dfInput[dtColumn+"_cat"] =dfInput[dtColumn]  

            #Assigning values to the coresponding column values
            for index, dtavgColumn in dfavg.iterrows() :

                dfInput = dfInput.replace(
                    {dtColumn: {dtavgColumn[dtColumn] :
                    dtavgColumn['row_num']}}
                    )                     
                        
    return dfInput


######################################################################################################


 # Calculating the expected value 

def expectation_function(y, x_value):

    # Mean value of each variable
    return np.mean(y[X == x_value]) 

######################################################################################################

 # Compare expected values with others
  
def compexpvalue(dfGroup,trgtVariable,infVariable):

    for i in range(len(dfGroup)):
        for j in range(len(dfGroup)):
        
             # Probability of X = v_i
            prob_X_i = np.mean(dfGroup == dfGroup[i][infVariable])
            
             # Probability of X = v_j
            prob_X_j = np.mean(dfGroup == dfGroup[j][infVariable]) 

             # Expected value calculation.
            expectation_ratio = abs(
                expectation_function(
                    dfGroup[trgtVariable], 
                    dfGroup[j][infVariable]
                ) /
                expectation_function(
                    dfGroup[trgtVariable],
                    dfGroup[i][infVariable]
                )
            )
    
            impact += prob_X_i * prob_X_j * expectation_ratio

    print ("GROUP "+[i]+": " +impact)

######################################################################################################    

    # C-IA Measure
    # Numerical-to-Categorical Coercion
def fciameasure(dfInput, infVariable, trgtVariable):

    # numerical checking 
    if 'int64' in list(dfInput.dtypes) :
    
        for dtColumn in dfInput :
        
            if (dfInput[dtColumn].dtypes == 'int64' or dfInput[dtColumn].dtypes == 'float64')  and dtColumn in infVariable: 
                
                minMAxDiff=dfInput[dtColumn].max() - dfInput[dtColumn].min() # Min Max Difference

                if(dfInput[dtColumn].min()==0):                
                    minMAxDiff=minMAxDiff/5
                    
                else :
                    minMAxDiff=minMAxDiff/4
            
                for i in range(5): 
                
                    if(i==0):

                        dfInput[dtColumn] = np.where(
                            (dfInput[dtColumn] <= minMAxDiff * (i+1)), 
                            i, 
                            dfInput[dtColumn]
                        )


                    else :
                        dfInput[dtColumn] = np.where(
                            (dfInput[dtColumn] > minMAxDiff * (i) &
                            dfInput[dtColumn] <=minMAxDiff * (i+1)), 
                            i, 
                            dfInput[dtColumn]
                        )
    dfGroup = dfInput.groupby(infVariable)
    # Calling expecetd value comparison function
    compexpvalue(dfGroup,trgtVariable,infVariable) 
    
    return 0

######################################################################################################

#Oaxaca-Blinder decomposition
    
def fobd(dfInput,infVariable,trgtVariable,groupCol):

    # Categorical to numerical Coercion
    dfInput = ca_coe(dfInput,infVariable,trgtVariable)    
    model=[]
    xmean = []
    ymean = []
    dfGroup=[]

    if (groupCol !=''):   
        for index,i in dfInput.groupby(groupCol) :      

            x = sm.add_constant(i[infVariable])
            model.append(
                sm.OLS(i[trgtVariable], 
                x.astype(float)).fit()
            )
            xmean.append(x.mean())

            ymean.append((i[trgtVariable]).mean())
        
        # Calling Oaxaca-Blinder decomposition calcualation function
        fobdCmp(model,xmean,ymean) 

    return 0    

######################################################################################################

    # Calculating Oaxaca-Blinder decomposition 
    
def fobdCmp(model,xmean,ymean) :
    
    # Calculating explanined part
    explained = (xmean[0] - xmean[1]).dot((model[0].params + model[1].params) / 2)

    # Calculating unexplained part
    unexplained = (model[0].params - model[1].params).dot((xmean[0] + xmean[1]) / 2)

    # Calculating total difference 
    total_difference = ymean[0] - ymean[1]
    
    print("Difference in coefficients:", total_difference)
    print("explained:", explained)
    print("unexplained:", unexplained)

    return 0    

######################################################################################################

 # Linear-Regression based adjustment

def fregAdj (dfInput, infVariable, trgtVariable):

    # Categorical to numerical Coercion
    dfInput = ca_coe(dfInput,infVariable,trgtVariable)

    for dtColumn in infVariable :

        x = sm.add_constant(dfInput[dtColumn]) # adding a constant
        model = sm.OLS(dfInput[trgtVariable], x).fit()
        predictions = model.predict(x) 

        # Calling print function 
        fprint_reg(
            model.params, 
            (x.drop(['const'], 
            axis=1)).keys(), 
            trgtVariable
        ) 
        
    return model   

    # Printing Linear-Regression results

def fprint_reg(model, infVariable, trgtVariable):

    # Printing the constent
    print(trgtVariable[0] + ' (Const) : ' + str(D(model['const'])))   

    # Printing the slops (Coefficients)
    for variableFields in infVariable :
        print(variableFields + ' : ' + str(D(model[variableFields])))

    print("\n")

######################################################################################################


# Ad-Hoc based adjustmet

def fadhmethod(dfInput, infVariable, trgtVariable):

    # numerical checking 
    if 'int64' in list(dfInput.dtypes) :
    
        for dtColumn in dfInput :
        
            if (dfInput[dtColumn].dtypes == 'int64' or dfInput[dtColumn].dtypes == 'float64')  and dtColumn in infVariable: 
                
                minMAxDiff=dfInput[dtColumn].max() -dfInput[dtColumn].min() # Min Max Difference

                if(dfInput[dtColumn].min()==0):                
                    minMAxDiff=minMAxDiff/5
                    
                else :
                    minMAxDiff=minMAxDiff/4
            
                for i in range(5): 
                
                    if(i==0):

                        dfInput[dtColumn] = np.where(
                            (dfInput[dtColumn] <= minMAxDiff * (i+1)), 
                            i, 
                            dfInput[dtColumn]
                        )


                    else :
                        dfInput[dtColumn] = np.where(
                            (dfInput[dtColumn] > minMAxDiff * (i) &
                            dfInput[dtColumn] <=minMAxDiff * (i+1)), 
                            i, 
                            dfInput[dtColumn]
                        )     
                   
    grouped = dfInput.groupby(infVariable)
    
    # Calculate conditional expectation except primary influencing factor
    conditional_means = grouped[trgtVariable].mean().reset_index(name='E_Y_given_X_Z')
    
    # Calculate marginal probability
    z_group_counts = dfInput.groupby(infVariable[1:]).size()
    z_marginal_probabilities = z_group_counts / len(dfInput)
    z_marginal_probabilities = z_marginal_probabilities.reset_index(name='P_Z')
    
    # Merge conditional expectations with marginal probabilities
    merged_data = conditional_means.merge(
                    z_marginal_probabilities, 
                    on=trgtVariable
                 )
    
    # Calculate the weighted contribution for P(Y | do(X))
    merged_data['Weighted_Contribution'] = (
                    merged_data['E_Y_given_X_Z'] 
                    * merged_data['P_Z']
            )
    
    # Sum over Z to get P(Y | do(X)) for each level of X
    results = merged_data.groupby(X)['Weighted_Contribution'].sum().reset_index(name='P_Y_do_X')
    
    return results

######################################################################################################

 # Main function

def main():
    
    pd.set_option("future.no_silent_downcasting", True)

    # excel file path
    mstrfle = pd.ExcelFile("file path") 

    # reading the master sheet
    excelMaster = pd.read_excel(mstrfle, 'MasterSheet') 


    # dataframe creation according to master sheet 
    dfMaster=pd.DataFrame(excelMaster, 
            columns= [
                        'sheetName',
                        'targetVariable',
                        'variableFields',
                        'stratifyVariable',
                        'groupCol'
                    ]
            ) 

    # removing null values
    dfMaster = dfMaster.fillna({'stratifyVariable': ''}) 
    dfMaster = dfMaster.fillna({'groupCol': ''}) 
    dfMaster = dfMaster.reset_index()

    excelReaderold=""
    
    try:
        for index,dfRows in dfMaster.iterrows(): 
            
            #  reading data sheet
            excelReader = pd.read_excel(mstrfle, dfRows['sheetName']) 

            # influencing variables
            infVariable=list(dfRows['variableFields'].split(",")) 

            # target variable
            trgtVariable=list(dfRows['targetVariable'].split(",")) 

            # Grouping variable
            groupCol =list(dfRows['groupCol'].split(",")) 

            dfInput= pd.DataFrame(
                        excelReader, 
                        columns= (
                            infVariable 
                            + 
                            trgtVariable
                        )
                    ) 
            
            if excelReaderold != dfRows['sheetName']:
                print (dfRows['sheetName'])

            print('Influencing variables : ' + dfRows['variableFields'])
            print('Target variables : ' + dfRows['targetVariable'])
            print('Stratify variables : ' + dfRows['stratifyVariable'])
            print('Grouping variables : ' + dfRows['groupCol'])


            # Calling Linear-Regression based adjustment function 
            fregAdj (dfInput,infVariable,trgtVariable)

            # Calling Oaxaca-Blinder decomposition function
            fobd(dfInput,infVariable,trgtVariable,groupCol)

            # Calling Ad-Hoc based adjustmet function
            fadhmethod(dfInput, infVariable, trgtVariable)

            # Calling C-IA Measure function
            fciameasure(dfInput, infVariable, trgtVariable)
    
    except NameError:
        print(NameError)
    
main()