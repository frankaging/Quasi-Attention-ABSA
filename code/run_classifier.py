import argparse

from util.train_helper import *

def run(args):

    device, n_gpu, output_log_file= system_setups(args)

    # data loader, we load the model and corresponding training and testing sets
    model, optimizer, train_dataloader, test_dataloader = \
        data_and_model_loader(device, n_gpu, args)

    # main training step    
    global_step = 0
    global_best_acc = -1
    epoch=0
    evaluate_interval = args.evaluate_interval
    # training epoch to eval
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        # train a teacher model solving this task
        global_step, global_best_acc = \
            step_train(train_dataloader, test_dataloader, model, optimizer, 
                        device, n_gpu, evaluate_interval, global_step, 
                        output_log_file, epoch, global_best_acc, args)
        epoch += 1

    logger.info("***** Global best performance *****")
    logger.info("accuracy on dev set: " + str(global_best_acc))

if __name__ == "__main__":
    from util.args_parser import parser
    args = parser.parse_args()
    run(args)