from importlib import import_module

class TrainerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(trainername, model, args):
        module = import_module(f'trainer.{trainername}')
        return module.Trainer(model=model, args=args)

class GenericTrainer:
    '''
    Base class for retriever; to implement a new retriever, inherit from this.
    '''
    group_idx ={
        'gender' : 0,
        'age' : 1,
        'race' : 2
    }
    group_dic = {
        'gender' : ['male', 'female'],
        'age' : ['young', 'old'],
        'race' : ['East Asian', 'White', 'Latino_Hispanic', 'Southeast Asian', 'Black', 'Indian', 'Middle Eastern']
    }
    def __init__(self, model, args):
        self.args = args
        self.model = model
        