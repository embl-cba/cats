package de.embl.cba.cats.utils;

public abstract class StringUtils
{
    public static int[] delimitedStringToIntegerArray(String s, String delimiter)
    {
        String[] sA = s.split(delimiter);
        int[] nums = new int[sA.length];

        for(int i = 0; i < nums.length; ++i)
        {
            nums[i] = Integer.parseInt(sA[i].trim());
        }

        return nums;
    }
}
