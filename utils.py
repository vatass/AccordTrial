'''
Utils 

'''
import torch 
import numpy as np

##### LOAD AND SAVE MODEL FUNCTIONS #### 

# Save the model state dictionary and the optimizer state and the data 
def save_model(model, optimizer, likelihood, filename="model_state.pth", train_x=None, train_y=None):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'likelihood_state_dict': likelihood.state_dict()
    }
    
    # Add training data if provided
    if train_x is not None:
        save_dict['train_x'] = train_x
    if train_y is not None:
        save_dict['train_y'] = train_y
    
    torch.save(save_dict, filename)


# Load the model, optimizer, and likelihood from the saved file
def load_model(model, optimizer, likelihood, filename="deep_kernel_gp_model.pth"):
    print('Load Model')
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    return model, optimizer, likelihood

##### PROCESS DATA FUNCTIONS #### 


def process_temporal_singletask_data(train_x, train_y, test_x, test_y, test_ids): 
    
    assert train_x.shape[0] == train_y.shape[0]

    train_x_data = [] 
    assert len(train_x) > 0

    for i, t in enumerate(train_x): 
        a = t.strip('][').split(', ')
        # print(a)
        b = [float(i) for i in a]    
        # print(b)
        train_x_data.append(np.expand_dims(np.array(b), 0))
    # print(train_x_data[0])


    test_x_data = []
    for i, t in enumerate(test_x): 
        a = t.strip('][').split(', ')
        b = [float(i) for i in a]
        test_x_data.append(np.expand_dims(np.array(b), 0))

    test_y_data = [] 
    # print(test_y)
    for i, t in enumerate(test_y): 
        # print(t)
        a = t.strip('][').split(', ')
        # print(a)
        b = [float(i) for i in a]
        # print(len(b))
        test_y_data.append(np.expand_dims(np.array(b), 0))
    
    train_y_data = [] 
    for i, t in enumerate(train_y): 
        a = t.strip('][').split(', ')
        b = [float(i) for i in a]
        # print(len(b))
        train_y_data.append(np.expand_dims(np.array(b), 0))

    train_x_data = np.concatenate(train_x_data, axis=0)
    test_x_data = np.concatenate(test_x_data, axis=0)
    train_y_data = np.concatenate(train_y_data, axis=0)
    test_y_data = np.concatenate(test_y_data, axis=0)
    
    train_y, test_y = np.array(train_y), np.array(test_y)

    # print(train_x_data)

    data_train_x = torch.Tensor(train_x_data)
    data_train_y = torch.Tensor(train_y_data)
    data_test_x = torch.Tensor(test_x_data)
    data_test_y = torch.Tensor(test_y_data)  

    # data_train_x,  data_train_y, data_test_x, data_test_y  = torch.Tensor(train_x_data), torch.Tensor(train_y_data), torch.Tensor(test_x_data), torch.Tensor(test_y_data)

    return data_train_x, data_train_y, data_test_x, data_test_y



def calc_coverage(predictions, groundtruth, intervals, per_task=True):
    '''
    predictions: list with predictions 
    groundtruth: list with true values 
    intervals: if list has two elements, then it is the upper and lower from GP-like models 
                if has more than two, then it is the same length as the predictions and groundtruth and comes from 
                the conformal algorithm  
    
    '''
    predictions_tensor = torch.Tensor(predictions)
    groundtruth_tensor = torch.Tensor(groundtruth)

    mean_coverage, mean_intervals = 0,0 

    if len(intervals) == 2: 
        # upper and lower 
        lower = intervals[0]
        upper = intervals[1]

        assert len(upper) == len(lower)
        groundtruth_tensor = torch.Tensor(groundtruth)
        upper_tensor = torch.Tensor(upper) 
        lower_tensor = torch.Tensor(lower)
        intervals = torch.abs(upper_tensor-lower_tensor)

        coverage =  torch.logical_and(upper_tensor >= groundtruth_tensor, lower_tensor <= groundtruth_tensor)
        # print('Coverage', coverage.shape)
        # print('Intervals', intervals.shape)
        # print('Coverage', coverage)

    else: 
        coverage = [] 
        upper, lower = [], [] 

        for i in range(len(intervals)):
            upper.append(predictions[i] + intervals[i])
            lower.append(predictions[i] - intervals[i])

        upper_tensor = torch.Tensor(upper) 
        lower_tensor = torch.Tensor(lower)

        intervals = torch.abs(lower_tensor-upper_tensor) 
        coverage =  torch.logical_and(upper_tensor >= groundtruth_tensor, lower_tensor <= groundtruth_tensor)
        # print('Coverage', coverage.shape)
        # print('Intervals', intervals.shape)

    mean_coverage = torch.count_nonzero(coverage)/coverage.shape[0]
    mean_intervals = torch.mean(intervals)
    return coverage, intervals, mean_coverage, mean_intervals
