from rfdetr import RFDETRNano

model = RFDETRNano()

model.train(
    dataset_dir='./dataset',
    epochs=10,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='./trained_model'
)
