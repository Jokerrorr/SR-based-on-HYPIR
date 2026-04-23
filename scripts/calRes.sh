python scripts/calRes.py \
--path_to_ori data/test/RealSRVal_crop128/test_HR \
--path_to_sr results/RealSRVal_crop128/full_bid/final \
--output results/RealSRVal_crop128/bid.csv \
--batch_size 8 \
--num_workers 4 \
--all true \
--fidonly false