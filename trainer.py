import time

import torch
from einops import rearrange
from tqdm import tqdm
from evaluator import evaluate
from batch_generater import cf_train_quadkey
from utils import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(model, max_len, train_data, train_sampler, train_bsz, train_num_neg, num_epoch, quadkey_processor, loc2quadkey,
          eval_data, eval_sampler, eval_bsz, eval_num_neg, optimizer, loss_fn, device, num_workers, log_path, result_path,
          id2gps,model_path,finished_epoch,dis_loss):
    writer=SummaryWriter("loss//matrix-spatial")
    for epoch_idx in range(num_epoch):
        start_time = time.time()
        running_loss = 0.
        processed_batch = 0.
        data_loader = DataLoader(train_data,
                                 sampler=LadderSampler(train_data, train_bsz),
                                 num_workers=num_workers, batch_size=train_bsz,
                                 collate_fn=lambda e: cf_train_quadkey(
                                     e,
                                     train_data,
                                     max_len,
                                     train_sampler,
                                     quadkey_processor,
                                     loc2quadkey,
                                     train_num_neg))
        print("=====epoch {:>2d}=====".format(epoch_idx))
        batch_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)

        model.train()
        # for batch_idx, (src_locs_, src_quadkeys_, src_times_, t_mat_, g_mat_, trg_locs_, trg_quadkeys_, data_size,src_periods_) in batch_iterator:
        # for batch_idx, (src_locs_, src_quadkeys_, src_times_, trg_locs_, trg_quadkeys_, data_size) in batch_iterator:
        for batch_idx, (src_locs_, src_quadkeys_, src_times_, w_mat, c_mat, trg_locs_, trg_quadkeys_, data_size,src_periods_) in batch_iterator:
            src_loc = src_locs_.to(device)
            src_quadkey = src_quadkeys_.to(device)
            src_time = src_times_.to(device)

            src_period = src_periods_.to(device)

            # t_mat = t_mat_.to(device)
            # g_mat = g_mat_.to(device)
            w_mat = w_mat.to(device) 
            c_mat = c_mat.to(device)
            trg_loc = trg_locs_.to(device)
            trg_quadkey = trg_quadkeys_.to(device)
            pad_mask = get_pad_mask(data_size, max_len, device)
            attn_mask = get_attn_mask(max_len, device)
            mem_mask = get_mem_mask(max_len, train_num_neg, device)
            key_pad_mask = get_key_pad_mask(data_size, max_len, train_num_neg, device)
            optimizer.zero_grad()
            # output,src_out = model(src_loc, src_quadkey, src_time, t_mat, g_mat, pad_mask, attn_mask,
            #                trg_loc, trg_quadkey, key_pad_mask, mem_mask, data_size,src_period)
            # output ,src_out= model(src_loc, src_quadkey, src_time, pad_mask, attn_mask,
            #                trg_loc, trg_quadkey, key_pad_mask, mem_mask, data_size)
            output ,src_out= model(src_loc, src_quadkey, src_time, pad_mask, attn_mask,
                           trg_loc, trg_quadkey, key_pad_mask, mem_mask, data_size,
                           src_period, w_mat, c_mat)
           
            output_forloss=src_out
            sorted=output_forloss.max(dim=-1,keepdim=True)
            # print(sorted[0])
            prob, order = output_forloss.topk(k=1, dim=-1, largest=True)
            trg_idx = rearrange(rearrange(trg_loc, 'b (k n) -> b k n', k=1 + train_num_neg), 'b k n -> b n k')
            pos_idx,neg_idx = trg_idx.split([1, train_num_neg], -1)
            dis_for_loss=dis_loss(order,pos_idx,id2gps,prob)
            
            output = rearrange(rearrange(output, 'b (k n) -> b k n', k=1 + train_num_neg), 'b k n -> b n k')
            pos_scores, neg_scores = output.split([1, train_num_neg], -1)
            loss = loss_fn(pos_scores, neg_scores,id2gps)
            keep = [torch.ones(e, dtype=torch.float32).to(device) for e in data_size]
            keep = fix_length(keep, 1, max_len, dtype="exclude padding term")

            sampe_loss = torch.sum(loss * keep) / torch.sum(torch.tensor(data_size).to(device))
            loss=0.01*dis_for_loss+sampe_loss
            
            loss.backward()
            optimizer.step()
            if(batch_idx%100==0):
                writer.add_scalar("loss",loss.item(),batch_idx+epoch_idx*batch_iterator.total)
                writer.add_scalar("sampe_loss",sampe_loss.item(),batch_idx+epoch_idx*batch_iterator.total)
                writer.add_scalar("dis_for_loss",dis_for_loss.item(),batch_idx+epoch_idx*batch_iterator.total)
            running_loss += loss.item()
            processed_batch += 1
            batch_iterator.set_postfix_str(f"loss={loss.item():.4f}, sample_loss={sampe_loss.item():4f}, dis_for_loss={dis_for_loss.item():4f}")

        epoch_time = time.time() - start_time
        cur_avg_loss = running_loss / processed_batch
        f = open(log_path, 'a+')
        print("time taken: {:.2f} sec".format(epoch_time))
        print("avg. loss: {:.4f}".format(running_loss / processed_batch))
        print("epoch={:d}, loss={:.4f}".format(epoch_idx + 1+finished_epoch, cur_avg_loss), file=f)
        f.close()
        if((epoch_idx+1)%5==0):
            torch.save(model.state_dict(), model_path+str(epoch_idx+finished_epoch)+".pth")
    

    print("training completed!")
    print("")
    print("=====evaluation under sampled metric (100 nearest un-visited locations)=====")
    hr, ndcg = evaluate(model, max_len, eval_data, eval_sampler, eval_bsz, eval_num_neg, quadkey_processor, loc2quadkey, device, num_workers)
    print("Hit@1: {:.4f}, NDCG@1: {:.4f}, Hit@5: {:.4f}, NDCG@5: {:.4f}, Hit@10: {:.4f}, NDCG@10: {:.4f} ".format(hr[0], ndcg[0],hr[4], ndcg[4], hr[9], ndcg[9]))
    f = open(result_path, 'a+')
    print("Hit@1: {:.4f}, NDCG@1: {:.4f}, Hit@5: {:.4f}, NDCG@5: {:.4f}, Hit@10: {:.4f}, NDCG@10: {:.4f} ".format(hr[0], ndcg[0],hr[4], ndcg[4], hr[9], ndcg[9]), file=f)
    f.close()
    writer.close()