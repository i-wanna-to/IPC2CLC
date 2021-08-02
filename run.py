from mapping_model import MappingModel

fofo = MappingModel(model_args_file='./model_args.json')
for k in range(fofo.args.kflod):
    model_path = fofo.train(kfold=k)
    fofo.predict(kfold=k, model_path=model_path)