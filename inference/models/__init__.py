def get_network(network_name):
    network_name = network_name.lower()
    # Original GR-ConvNet
    if network_name == 'graspmamba':
        from .graspmamba import Grasp_Mamba
        return Grasp_Mamba
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
