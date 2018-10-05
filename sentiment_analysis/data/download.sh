baidu="baidu.zip","1hzkg3zmClZQcjVPSZOAlXRtogAkmq-pb"
hotel="hotel_sent.zip","11mu6rgPtaivmA-3kWY6TqKG21UI4-D-U"
ptt="ptt.zip","1sBD-ttTLfjqFgxMMBGoyyb0cNErhRS8f"
weiboPos="weibo_pos.txt","1FAmIppE7jzjRhuN2ClRxVAUt4P6Nsm2U"
weiboNeg="weibo_neg.txt","1zX6afbtkSdtPVWQWKjxqZWST8vXVSrGP"

files_arr=($baidu $hotel $ptt $weiboPos $weiboNeg)
for f in ${files_arr[@]}; do
    IFS=',' read filename fileid <<< "${f}"
    curl -L -o "${filename}" "https://drive.google.com/uc?export=download&id=${fileid}"
    echo "${filename}" ":" "${fileid} downloaded!"
    if [[ ${filename} = *"zip"* ]]; then
        unzip ${filename}
        rm ${filename}
    elif [[ ${filename} = *"gz"* ]]; then
        gunzip ${filename}
        rm ${filename}
    elif [[ ${filename} = *".tar.bz"* ]]; then
        tar jxvf ${filename}
        rm ${filename}
    elif [[ ${filename} = *".tar.gz"* ]]; then
        tar zxvf ${filename}
        rm ${filename}
    fi
done


