def get_network(network_name):
    network_name = network_name.lower()
    # Original GR-ConvNet
    if network_name == 'graspmamba':
        from .graspmamba import Grasp_Mamba
        return Grasp_Mamba
    elif network_name == 'graspmambav2':
        from .graspmambav2 import Grasp_Mamba
        return Grasp_Mamba
    elif network_name == 'graspmambav3':
        from .graspmambav3 import Grasp_Mamba
        return Grasp_Mamba
    elif network_name == 'graspmambav4':
        from .graspmambav4 import Grasp_Mamba
        return Grasp_Mamba
    elif network_name == 'graspmambav5':
        from .graspmambav5 import Grasp_Mamba
        return Grasp_Mamba
    elif network_name == 'graspmambav6':
        from .graspmambav6 import Grasp_Mamba
        return Grasp_Mamba
    elif network_name == 'graspmambav7':
        from .graspmambav7 import Grasp_Mamba
        return Grasp_Mamba
    elif network_name == 'graspmambav8':
        from .graspmambav8 import Grasp_Mamba
        return Grasp_Mamba
    elif network_name == 'graspmambav7_bert':
        from .graspmambav7_bert import Grasp_Mamba
        return Grasp_Mamba
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
