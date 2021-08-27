from core.data_provider import kth_action, mnist, bair

datasets_map = {
    'mnist': mnist,
    'action': kth_action,
    'bair': bair,
}


def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  img_width, seq_length, injection_action, is_training=True):
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')
    if dataset_name == 'mnist':
        test_input_param = {'paths': valid_data_list,
                            'minibatch_size': batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name + 'test iterator'}
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {'paths': train_data_list,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'is_output_sequence': True,
                                 'name': dataset_name + ' train iterator'}
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param)
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle

    if dataset_name == 'action':
        input_param = {'paths': valid_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'input_data_type': 'float32',
                       'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handle
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle

    if dataset_name == 'bair':
        test_input_param = {'valid_data_paths': valid_data_list,
                            'train_data_paths': train_data_list,
                            'batch_size': batch_size,
                            'image_width': img_width,
                            'image_height': img_width,
                            'seq_length': seq_length,
                            'injection_action': injection_action,
                            'input_data_type': 'float32',
                            'name': dataset_name + 'test iterator'}
        input_handle_test = datasets_map[dataset_name].DataProcess(test_input_param)
        test_input_handle = input_handle_test.get_test_input_handle()
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {'valid_data_paths': valid_data_list,
                                 'train_data_paths': train_data_list,
                                 'image_width': img_width,
                                 'image_height': img_width,
                                 'batch_size': batch_size,
                                 'seq_length': seq_length,
                                 'injection_action': injection_action,
                                 'input_data_type': 'float32',
                                 'name': dataset_name + ' train iterator'}
            input_handle_train = datasets_map[dataset_name].DataProcess(train_input_param)
            train_input_handle = input_handle_train.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle