# define lr_scheduler
def Poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=100, power=0.9):
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer
    
    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr
