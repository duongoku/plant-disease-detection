import { cn } from "@/lib/utils";
import { ButtonHTMLAttributes } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: "default" | "primary" | "outline";
}

export function Button({ variant = "default", className, ...props }: ButtonProps) {
    return (
        <button
            className={cn(
                "px-4 py-2 rounded-lg text-white transition-all duration-300",
                variant === "primary" && "bg-green-600 hover:bg-green-700",
                variant === "outline" && "border border-green-600 text-green-600 hover:bg-green-100",
                className
            )}
            {...props}
        />
    );
}
