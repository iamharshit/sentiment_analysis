declare -i count
count=0

declare -i total
total=0
cat a.txt | while read p; do
   total=total+1
   if echo "$p" | grep '*'; then
	   echo '-----------------'
   else
	   echo $p
	   var=`find b.txt -type f -print | xargs grep -- '$p'` #"`grep $p glove_twitter.txt`"
	   if [ -z "$var" ]
	    then :
	   else
	    echo "$p" >> myfile2.txt
	    count=count+1
	   fi
    fi
done

echo $total
echo $count
