#!/bin/bash
i=4
j=$[(i-1)%12]
echo $j
str1="abcde123"
sub="123"
echo ${str1%${sub}}


