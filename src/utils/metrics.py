def confusion_matrix_binary(t_model, t_data):
    conf_mat = [[0, 0], [0, 0]]

    for i in range(t_model.shape[0]):
        if(t_model[i] == 1 and t_data[i] == 1 ):
            conf_mat[0][0] = conf_mat[0][0] + 1
        
        if(t_model[i] == 0 and t_data[i] == 0 ):
            conf_mat[1][1] = conf_mat[1][1] + 1

        if(t_model[i] == 0 and t_data[i] == 1 ):
            conf_mat[1][0] = conf_mat[1][0] + 1

        if(t_model[i] == 1 and t_data[i] == 0 ):
            conf_mat[0][1] = conf_mat[0][1] + 1
    
    return conf_mat